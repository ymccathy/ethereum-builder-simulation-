
# this script simulates a repeated PD-style game among Ethereum block builders
# under Proposer–Builder Separation (PBS). It supports two abstractions:
#   Variant A (aggregate PD): all builders announce C/D each slot; if any choose D,
#      the slot outcome is D (overfill). If all choose C, outcome is C (under/target).
#   Variant B (winner-acts PD): only the winning builder (sampled by win shares π_i)
#     chooses C/D; others observe a public (possibly noisy) signal and update punishments.
#
# You can plug in parameters for basefee dynamics (EIP-1559 stylized), valuation
# functions for public tips and private MEV, builder win-shares π, keep-fractions α,
# strategies (Grim, Generous-TFT, Always-D/C), punishment length, noise, etc.
#
# HOW TO USE
# 1) Adjust the CONFIG section at the bottom.
# 2) Run `simulate!(cfg)`; it returns a summary with averages you can print/plot.

using Statistics

#############################
# Types & Data Structures   #
#############################

abstract type Strategy end
struct AlwaysC <: Strategy end
struct AlwaysD <: Strategy end

struct Grim <: Strategy end                 # punish forever after a detected D

struct GenerousTFT <: Strategy
    forgive_prob::Float64                   # probability to forgive after punishing one step
end

struct Builder
    name::String
    pi::Float64         # win probability π_i (sum π_i = 1)
    alpha::Float64      # keep fraction α_i(B) ~ simplified as constant
    theta_mu::Float64   # mean of θ (private orderflow advantage) in [0,1]
    theta_sigma::Float64# σ of θ (truncated normal approx in [0,1])
    strat::Strategy
end

struct ModelParams
    # Basefee dynamics (stylized EIP-1559)
    kappa::Float64      # responsiveness ~ 0.125 per full overfill
    Tgas::Float64       # target gas (normalized = 1.0). Actions choose g as a multiple of Tgas
    actions::Vector{Float64}  # gas multipliers, e.g., [0.8, 1.0, 1.2]

    # Valuation params
    Dmax::Float64       # max public Willingness TO Pay WTP (in "gwei" units scale for effect)
    beta_by_action::Vector{Float64}  # scales for public tips per action index
    cpriv_by_action::Vector{Float64} # private MEV base per action index

    # Repeated-game & monitoring
    delta::Float64      # discount factor for reporting; simulation uses finite horizon
    noise_eps::Float64  # P(bad signal | C) false positive; P(good | D) = false negative
    punishment_L::Int   # finite punishment length for non-Grim (L=∞ approximated via Grim)

    # Mode flags
    winner_acts::Bool   # true => Variant B (only winner acts), false => Variant A (aggregate rule)
endbn

mutable struct SimState
    B::Float64                # current base fee state
    in_punishment::Bool       # public punishment flag (for aggregate-mode) or shared flag
    punish_counter::Int       # remaining punishment steps (if using finite L)
    last_outcome_D::Bool      # last observed outcome was D (for TFT-like behavior)

    # accounting
    t::Int
    coop_count::Int
    over_count::Int
    profits::Dict{String,Float64}
    outcome_history::Vector{Bool}  # true if D, false if C
    B_history::Vector{Float64}
end

#############################
# Helpers                   #
#############################

# Truncated normal in [0,1] via simple clipping around N(mu, sigma)
rand_theta(mu, sigma) = clamp(mu + sigma*randn(), 0.0, 1.0)

# Public tips valuation: V_pub(B, action_idx) = beta(a) * max(0, Dmax - B)
function V_pub(mp::ModelParams, B::Float64, aidx::Int)
    return mp.beta_by_action[aidx] * max(0.0, mp.Dmax - B)
end

# Private MEV valuation: V_priv(action_idx, θ) = θ * cpriv(a)
V_priv(mp::ModelParams, aidx::Int, θ::Float64) = θ * mp.cpriv_by_action[aidx]

# Gross block value: V = V_pub + V_priv
function V_gross(mp::ModelParams, B::Float64, aidx::Int, θ::Float64)
    return V_pub(mp,B,aidx) + V_priv(mp,aidx,θ)
end

# aidx base fee under chosen gas multiplier g = actions[aidx]
function next_basefee(mp::ModelParams, B::Float64, aidx::Int)
    g = mp.actions[aidx]
    return B * (1.0 + mp.kappa * (g - mp.Tgas) / mp.Tgas)
end

# Public signal of outcome with noise: return true if signal says D (overfill), false if C
function noisy_signal_of_outcome(isD::Bool, eps::Float64)
    if isD
        # With false negative prob eps, we mistakenly see C
        return rand() > eps
    else
        # With false positive prob eps, we mistakenly see D
        return rand() < eps
    end
end

#############################
# Strategy Policies         #
#############################

# Map strategy + punishment state + last signal to intended action: true=>D, false=>C
function intended_isD(strat::Strategy, in_punishment::Bool, last_outcome_D::Bool)
    if strat isa AlwaysD
        return true
    elseif strat isa AlwaysC
        return in_punishment ? true : false  # if in punishment, force D
    elseif strat isa Grim
        return in_punishment ? true : false
    elseif strat isa GenerousTFT
        s = strat::GenerousTFT
        if in_punishment
            # punish this step with D, but maybe forgive
            if rand() < s.forgive_prob
                return false
            else
                return true
            end
        else
            # cooperate unless last observed outcome was D
            return last_outcome_D
        end
    else
        return true
    end
end

# Choose action index from intended C/D
# For simplicity: C -> action index closest to target; D -> highest gas multiplier
function action_from_CD(mp::ModelParams, isD::Bool)
    if isD
        return length(mp.actions)  # last index = max overfill
    else
        return 1
    end 
        # pick the index whose action is closest to Tgas from below or equal
        # diffs = abs.(mp.actions .- mp.Tgas)
        # prefer the one == Tgas if present, else smallest diff
        # idx = argmin(diffs)
        #return idx
    
    end


#############################
# Slot Simulation           #
#############################

# Compute winner’s realized net profit given builder, state, and action
function winner_profit(mp::ModelParams, b::Builder, B::Float64, aidx::Int)
    θ = rand_theta(b.theta_mu, b.theta_sigma)
    V = V_gross(mp, B, aidx, θ)
    return b.alpha * V
end

# Sample index of winning builder according to π
function sample_winner(builders::Vector{Builder})
    ps = [b.pi for b in builders]
    r = rand()
    acc = 0.0
    for (i,p) in enumerate(ps)
        acc += p
        if r <= acc
            return i
        end
    end
    return length(builders)  # fallback
end

#############################
# Main Simulation Loop      #
#############################
mutable struct Config
    builders::Vector{Builder}
    mp::ModelParams
    Tslots::Int
    B0::Float64
end

mutable struct SimSummary
    avg_B::Float64
    coop_rate::Float64   # fraction of slots with aggregate outcome C
    profits::Dict{String,Float64}
    D_fraction_by_builder::Dict{String,Float64}
end

function init_state(cfg::Config)
    return SimState(cfg.B0, false, 0, false, 0, 0, 0, Dict(b.name=>0.0 for b in cfg.builders), Bool[], Float64[])
end

# Simple bidding policy: use builder's alpha as baseline shading (β),
# add a tiny boost when not in punishment; clamp to [0.1, 0.99].
function choose_beta(b::Builder, st::SimState, mp::ModelParams)
    base = b.alpha
    boost = st.in_punishment ? 0.0 : 0.05
    return clamp(base + boost, 0.1, 0.99)
end


function simulate!(cfg::Config)
    st = init_state(cfg)
    mp = cfg.mp
    builders = cfg.builders
    # track intended D rates by builder (for reporting)
    D_count = Dict(b.name=>0 for b in builders)

    for t in 1:cfg.Tslots
        st.t = t

        # Determine intended actions (C/D) for each builder given current punishment state
        intended = Dict{String,Bool}()
        for b in builders
            d = intended_isD(b.strat, st.in_punishment, st.last_outcome_D)
            intended[b.name] = d
            if d; D_count[b.name] += 1; end
        end

        if mp.winner_acts
            # Variant B with an explicit first-price auction per slot 
            bids = Float64[]
            vals = Float64[]
            aidxs = Int[]
            names = String[]

            # Each builder forms an intended action, a gross value v_i, and a bid b_i = β_i * v_i
            for b in builders
                push!(names, b.name)
                aidx_i = action_from_CD(mp, intended[b.name])
                θ_i = rand_theta(b.theta_mu, b.theta_sigma)
                v_i = V_gross(mp, st.B, aidx_i, θ_i)
                β_i = choose_beta(b, st, mp)
                b_i = β_i * v_i
                push!(aidxs, aidx_i)
                push!(vals, v_i)
                push!(bids, b_i)
            end

            # Highest bid wins
            wi = argmax(bids)
            bw = builders[wi]
            v_w = vals[wi]
            b_w = bids[wi]
            aidx = aidxs[wi]

            # Winner's net profit = v_w - b_w (can't be negative)
            prof = max(v_w - b_w, 0.0)
            st.profits[bw.name] += prof

            # Update base fee based on winner's chosen action
            Bnext = next_basefee(mp, st.B, aidx)

            # Determine if the realized action is a "D" (overfill) for signaling/punishment
            isD = mp.actions[aidx] > mp.Tgas

            # Public (noisy) signal of D/C
            sigD = noisy_signal_of_outcome(isD, mp.noise_eps)

            # Punishment state machine
            if st.in_punishment
                if mp.punishment_L > 0
                    st.punish_counter -= 1
                    if st.punish_counter <= 0
                        st.in_punishment = false
                    end
                end
            else
                if sigD
                    st.in_punishment = true
                    st.punish_counter = mp.punishment_L
                end
            end

            # Bookkeeping & histories
            st.last_outcome_D = sigD
            push!(st.outcome_history, isD)
            push!(st.B_history, st.B)
            st.B = Bnext
            if isD; st.over_count += 1 else; st.coop_count += 1 end

            
            # Update punishment logic
            if st.in_punishment
                # decrement counter (if finite L). If Grim, we never exit.
                if mp.punishment_L > 0
                    st.punish_counter -= 1
                    if st.punish_counter <= 0
                        st.in_punishment = false
                    end
                end
            else
                if sigD
                    st.in_punishment = true
                    st.punish_counter = mp.punishment_L
                end
            end

            st.last_outcome_D = sigD
            push!(st.outcome_history, isD)
            push!(st.B_history, st.B)
            st.B = Bnext
            if isD; st.over_count += 1 else; st.coop_count += 1 end


        # Variant A: aggregate PD — if any D, outcome is D; else C
        else
            anyD = any(values(intended))
            aidx = action_from_CD(mp, anyD)

            # Profit distribution: either pay only a sampled winner, or expected profits
            # Here we do expected profits (clean PD mapping): each builder earns π_i * profit_if_winner
            for b in builders
                prof_w = winner_profit(mp, b, st.B, aidx)
                st.profits[b.name] += b.pi * prof_w
            end

            # Update base fee, signal, punishment (public)
            Bnext = next_basefee(mp, st.B, aidx)
            sigD = noisy_signal_of_outcome(anyD, mp.noise_eps)

            if st.in_punishment
                if mp.punishment_L > 0
                    st.punish_counter -= 1
                    if st.punish_counter <= 0
                        st.in_punishment = false
                    end
                end
            else
                if sigD
                    st.in_punishment = true
                    st.punish_counter = mp.punishment_L
                end
            end

            st.last_outcome_D = sigD
            push!(st.outcome_history, anyD)
            push!(st.B_history, st.B)
            st.B = Bnext
            if anyD; st.over_count += 1 else; st.coop_count += 1 end
        end
    end

    
    # Build summary
    avg_B = mean(st.B_history)
    coop_rate = st.coop_count / cfg.Tslots
    D_fraction_by_builder = Dict(b.name => D_count[b.name] / cfg.Tslots for b in builders)
    return SimSummary(avg_B, coop_rate, st.profits, D_fraction_by_builder)
end

#############################
# CONFIG (edit these)       #
#############################

# Example: 3 actions = [underfill, target, overfill]
actions = [0.8, 1.0, 1.2]          # gas multipliers g/T
beta_by_action = [0.8, 1.0, 1.2]    # scale of public tips per action
cpriv_by_action = [1.0, 2.0, 3.0]   # private MEV base per action

mp = ModelParams(
    0.125,           # kappa
    1.0,             # Tgas
    actions,
    10.0,            # Dmax
    beta_by_action,
    cpriv_by_action,
    0.97,            # delta
    0.00,            # noise_eps (5% misclassification)
    5,               # punishment_L (slots)
    false             # winner_acts (Variant B). Set false for aggregate PD (Variant A).
)

# Builders: name, π, α, θμ, θσ, strategy
builders = [
    Builder("Strong", 0.35, 0.25, 0.7, 0.15, GenerousTFT(0.15)),
    Builder("Mid",    0.25, 0.25, 0.5, 0.20, Grim()),
    Builder("WeakA",  0.20, 0.25, 0.3, 0.15, AlwaysD()),
    Builder("WeakB",  0.20, 0.25, 0.3, 0.15, AlwaysC()),
]

builders2 = [
    Builder("A",  0.5, 0.30, 0.3, 0.15, AlwaysD()),
    Builder("B",  0.5, 0.30, 0.3, 0.15, AlwaysD()),
]

builders3 = [
    Builder("1", 0.25, 0.45, 0.7, 0.15, GenerousTFT(0.15)),
    Builder("2",  0.25, 0.35, 0.5, 0.20, Grim()),
    Builder("3",  0.25, 0.30, 0.3, 0.15, AlwaysD()),
    Builder("4",  0.25, 0.30, 0.3, 0.15, AlwaysC()),
]

# Base Fee is not changing 
builders4 = [
    Builder("1", 0.25, 1, 0.7, 0.15, AlwaysC()),
]

builderstest = [
    Builder("titan1", 0.5, 0.45, 0.7, 0.15,  AlwaysD()),
    Builder("titan2",  0.5, 0.45, 0.7, 0.15, AlwaysD()),
    Builder("weak",  0.1, 0.1, 0.2, 0.15, AlwaysD()),
]

cfg = Config(builderstest, mp, 200, 5.0)  # Tslots=2000, initial basefee B0=5


# --- Identical Player --- 


# Build n identical builders for symmetric experiments
function make_identical_builders(n::Int; alpha::Float64, strat::Strategy=Grim(),
                                 theta_mu::Float64=0.5, theta_sigma::Float64=0.1)
    pi = 1.0 / n
    [Builder("B$(i)", pi, alpha, theta_mu, theta_sigma, strat) for i in 1:n]
end

# Run one scenario and print summary
function run_scenario(n::Int; alpha::Float64, noise::Float64, L::Int,
                      cpriv_overfill::Float64,
                      actions::Vector{Float64}=[0.8,1.0,1.2],
                      beta_by_action::Vector{Float64}=[0.8,1.0,1.2],
                      Dmax::Float64=10.0, kappa::Float64=0.125,
                      Tslots::Int=5000, B0::Float64=5.0,
                      strat::Strategy=Grim())
    # Temptation knob: private MEV for overfill (last index)
    cpriv = [1.0, 2.0, cpriv_overfill]

    mp_local = ModelParams(kappa, 1.0, actions, Dmax, beta_by_action, cpriv,
                           0.97, noise, L, true)  # winner_acts = true

    builders_local = make_identical_builders(n; alpha=alpha, strat=strat)
    cfg_local = Config(builders_local, mp_local, Tslots, B0)
    res = simulate!(cfg_local)
    println("n=$n  α≈$alpha  noise=$noise  L=$L  cpriv_overfill=$cpriv_overfill")
    println("  coop_rate  = ", round(res.coop_rate, digits=3))
    println("  avg_B      = ", round(res.avg_B, digits=3))
    println("  profits    = ", res.profits)
    println("  D_fraction = ", res.D_fraction_by_builder)
    println()
    return res
end

# Convenience sweeps for the two extreme cases
function sweep_symmetric()
    println("=== Symmetric, cooperative-friendly (low temptation) ===")
    for n in (2,4,8)
        # approximate “perfect competition” margin with α ≈ 1/n
        run_scenario(n; alpha=1/n, noise=0.0, L=20, cpriv_overfill=1.5, strat=Grim())
    end

    println("=== Symmetric, high-temptation stress test ===")
    for n in (2,4,8)
        run_scenario(n; alpha=1/n, noise=0.0, L=20, cpriv_overfill=6.0, strat=Grim())
    end
end


# Example run:
# using Statistics
# res = simulate!(cfg)
# println("avg_B         = ", res.avg_B)
# println("coop_rate     = ", round(res.coop_rate, digits=3)) ???!!! 
# println("profits       = ", res.profits)
# println("D_fraction_by_builder = ", res.D_fraction_by_builder)


# NOTES:
# - To model harsher punishment (Grim), set punishment_L to a very large number and/or use Grim for all.
# - To approximate APS-like continuity that favors multi-block strategies, raise Strong’s π and α, and lower noise_eps.
# - To compare Variant A vs B, flip `winner_acts` and observe changes in coop_rate and avg_B.