# Extension to your existing code: Mempool → Builder cartel dynamics

using Statistics
using Distributions  # For Poisson distribution

#############################
# New Types for Mempool Game #
#############################

struct Transaction
    id::Int
    priority_fee::Float64  # tip willing to pay (gwei)
    gas_units::Float64     # gas required
    arrival_time::Int      # which slot tx arrived
    patience::Float64      # drops out after waiting this many slots
end

mutable struct MempoolState
    txs::Vector{Transaction}           # pending transactions
    next_tx_id::Int
    demand_params::NamedTuple{(:λ_base, :γ, :fee_sensitivity), Tuple{Float64,Float64,Float64}}
end

struct BuilderAction
    fill_rate::Float64      # ∈ [0,1], what fraction of profitable txs to include
    target_gas::Float64     # gas to include (as multiple of Tgas)
end

#############################
# Demand & Transaction Generation #
#############################

"""
Demand model: arrival rate increases when base fee is low
λ(B) = λ_base * (1 + demand_boost / (B + 0.1))
This creates higher demand when B is low, with a floor to prevent explosion
"""
function demand_rate(B::Float64, params)
    # Much stronger demand response to low base fees
    B_floor = 0.1
    demand_boost = 2.0  # INCREASED from 0.1 to 2.0 for stronger effect
    base_demand = params.λ_base
    
    # Add penalty when B is very high to reduce demand
    high_fee_penalty = B > 10.0 ? exp(-0.05 * (B - 10.0)) : 1.0
    
    return base_demand * (1.0 + demand_boost / (B + B_floor)) * high_fee_penalty
end

"""
Generate new transactions arriving in this slot
Priority fees follow a distribution that depends on:
  - Base fee B (higher B → users bid higher tips to ensure inclusion)
  - Urgency distribution (some users need fast inclusion)
"""
function generate_new_txs!(mp::MempoolState, B::Float64, slot::Int)
    λ = demand_rate(B, mp.demand_params)
    n_new = round(Int, λ + sqrt(λ) * randn())  # Poisson approximation
    n_new = max(0, n_new)
    
    for _ in 1:n_new
        # Priority fee: mix of patient (low tip) and urgent (high tip) users
        # fee_sensitivity: how much users adjust tips based on current B
        base_tip = 0.5 + rand() * 4.0  # base range [0.5, 4.5] gwei - INCREASED
        urgency_boost = rand() < 0.3 ? rand() * 5.0 : 0.0  # 30% are urgent - INCREASED
        fee = base_tip + urgency_boost + mp.demand_params.fee_sensitivity * B
        
        tx = Transaction(
            mp.next_tx_id,
            fee,
            21000.0 + rand() * 200000.0,  # gas units [21k, 221k] - INCREASED variance
            slot,
            10.0 + rand() * 20.0  # patience: 10-30 slots - INCREASED patience
        )
        push!(mp.txs, tx)
        mp.next_tx_id += 1
    end
    
    # Remove impatient transactions that have waited too long
    filter!(tx -> (slot - tx.arrival_time) < tx.patience, mp.txs)
end

#############################
# Builder Block Construction #
#############################

"""
Builder selects transactions for block given:
  - Current mempool
  - Fill rate f (cartel strategy)
  - Base fee B
  - Target gas Tgas

Returns: (selected_txs, total_gas, total_tips, remaining_txs)
"""
function construct_block(
    mp::MempoolState, 
    B::Float64, 
    fill_rate::Float64, 
    Tgas::Float64,
    max_gas::Float64=30_000_000.0
)
    # Sort by priority fee (greedy selection)
    sorted_txs = sort(mp.txs, by=tx->tx.priority_fee, rev=true)
    
    # How much gas to target? fill_rate controls this
    target_gas_budget = fill_rate * Tgas
    
    selected = Transaction[]
    total_gas = 0.0
    total_tips = 0.0
    
    for tx in sorted_txs
        # Only include if profitable (priority fee > 0) and fits in budget
        if tx.priority_fee > 0 && total_gas + tx.gas_units <= min(target_gas_budget, max_gas)
            push!(selected, tx)
            total_gas += tx.gas_units
            total_tips += tx.priority_fee * tx.gas_units / 1e9  # convert to ETH scale
        end
    end
    
    # Remaining transactions stay in mempool
    remaining = setdiff(sorted_txs, selected)
    
    return (selected=selected, total_gas=total_gas, total_tips=total_tips, remaining=remaining)
end

#############################
# Cartel Simulation         #
#############################

"""
Two-player cartel game:
  - Both builders choose fill_rate ∈ [0.5, 1.0]
  - If both withhold (f < 1), cartel succeeds → base fee manipulation
  - If one deviates (f = 1), defector captures all withheld demand
  
Payoff structure similar to Prisoner's Dilemma:
  - (Cooperate, Cooperate): both benefit from sustained lower B
  - (Cooperate, Defect): defector gets huge gain, cooperator loses
  - (Defect, Defect): competitive outcome, no manipulation
"""
mutable struct CartelSimState
    mp::MempoolState
    B::Float64
    profits::Dict{String, Vector{Float64}}  # profit history per builder
    B_history::Vector{Float64}
    fill_history::Dict{String, Vector{Float64}}
    slot::Int
end

function init_cartel_sim(B0::Float64, λ_base::Float64=100.0)  # INCREASED from 50 to 100
    mp = MempoolState(
        Transaction[],
        1,
        (λ_base=λ_base, γ=0.15, fee_sensitivity=0.3)
    )
    
    # Seed with MORE initial transactions
    for i in 1:100  # INCREASED from 20 to 100
        tx = Transaction(i, 1.0 + rand()*3.0, 21000.0 + rand()*100000.0, 0, 20.0)
        push!(mp.txs, tx)
    end
    
    return CartelSimState(
        mp, B0,
        Dict("Builder1"=>Float64[], "Builder2"=>Float64[]),
        [B0],
        Dict("Builder1"=>Float64[], "Builder2"=>Float64[]),
        0
    )
end

"""
Simulate one slot with two builders choosing fill rates
winner_idx: which builder wins this slot (alternating or probabilistic)
"""
function simulate_cartel_slot!(
    st::CartelSimState,
    fill_rates::Dict{String, Float64},
    winner_name::String,
    Tgas::Float64=15_000_000.0,
    kappa::Float64=0.125
)
    st.slot += 1
    
    # Generate new transactions based on current base fee
    generate_new_txs!(st.mp, st.B, st.slot)
    
    # Winner constructs block with their chosen fill rate
    f_winner = fill_rates[winner_name]
    result = construct_block(st.mp, st.B, f_winner, Tgas)
    
    # Winner's profit (simplified: just tips, could add MEV)
    profit = result.total_tips
    push!(st.profits[winner_name], profit)
    
    # Loser gets 0 this slot
    loser_name = winner_name == "Builder1" ? "Builder2" : "Builder1"
    push!(st.profits[loser_name], 0.0)
    
    # Update fill rate history
    push!(st.fill_history[winner_name], f_winner)
    push!(st.fill_history[loser_name], fill_rates[loser_name])
    
    # Update mempool (remove selected txs)
    st.mp.txs = result.remaining
    
    # Update base fee via EIP-1559 with a floor
    gas_ratio = result.total_gas / Tgas
    B_new = st.B * (1.0 + kappa * (gas_ratio - 1.0))
    
    # Add base fee floor to prevent crash to zero (like real Ethereum)
    # But make it lower so we can see more variation
    B_floor = 0.1  # minimum 0.1 gwei (was 0.5)
    st.B = max(B_new, B_floor)
    
    push!(st.B_history, st.B)
    
    return result
end

#############################
# Experiment Scenarios      #
#############################

"""
Scenario 1: Both builders always cooperate (cartel holds)
Fill rate f = 0.8 (underfill by 20%)
"""
function run_stable_cartel(T_slots::Int=200)
    st = init_cartel_sim(5.0)
    
    for t in 1:T_slots
        winner = t % 2 == 0 ? "Builder1" : "Builder2"
        simulate_cartel_slot!(
            st,
            Dict("Builder1"=>0.8, "Builder2"=>0.8),
            winner
        )
    end
    
    return st
end

"""
Scenario 2: One builder defects (fills fully) while other cooperates
Tests if deviation is profitable
"""
function run_defection(T_slots::Int=200)
    st = init_cartel_sim(5.0)
    
    for t in 1:T_slots
        winner = t % 2 == 0 ? "Builder1" : "Builder2"
        # Builder1 defects (f=1.0), Builder2 cooperates (f=0.8)
        simulate_cartel_slot!(
            st,
            Dict("Builder1"=>1.0, "Builder2"=>0.8),
            winner
        )
    end
    
    return st
end

"""
Scenario 3: Both builders compete (no cartel)
Both fill fully (f=1.0)
"""
function run_competition(T_slots::Int=200)
    st = init_cartel_sim(5.0)
    
    for t in 1:T_slots
        winner = t % 2 == 0 ? "Builder1" : "Builder2"
        simulate_cartel_slot!(
            st,
            Dict("Builder1"=>1.0, "Builder2"=>1.0),
            winner
        )
    end
    
    return st
end

"""
Scenario 4: Dynamic demand shock
Test cartel profitability under varying demand conditions
"""
function run_demand_shock(T_slots::Int=300)
    st = init_cartel_sim(5.0, 30.0)  # start with low demand
    
    for t in 1:T_slots
        # Demand shock at t=100
        if t == 100
            st.mp.demand_params = (λ_base=100.0, γ=0.15, fee_sensitivity=0.3)
        end
        
        winner = t % 2 == 0 ? "Builder1" : "Builder2"
        # Cartel strategy: underfill more aggressively during low demand
        f = t < 100 ? 0.7 : 0.85
        simulate_cartel_slot!(
            st,
            Dict("Builder1"=>f, "Builder2"=>f),
            winner
        )
    end
    
    return st
end

#############################
# Analysis & Plotting       #
#############################

function analyze_results(st::CartelSimState)
    println("\n=== Cartel Simulation Results ===")
    println("Final base fee: ", round(st.B, digits=3), " gwei")
    println("Avg base fee:   ", round(mean(st.B_history), digits=3), " gwei")
    println("\nTotal profits:")
    for (name, profits) in st.profits
        total = sum(profits)
        println("  $name: ", round(total, digits=2), " ETH")
    end
    println("\nAvg fill rates:")
    for (name, fills) in st.fill_history
        avg_fill = mean(filter(!isnan, fills))
        println("  $name: ", round(avg_fill, digits=3))
    end
    println("\nMempool size at end: ", length(st.mp.txs), " pending txs")
end

#############################
# Run Experiments           #
#############################

# Example usage:
# Run stable cartel (both cooperate, f=0.8)
# Run all experiments
println("\n" * "="^50)
println("RUNNING CARTEL EXPERIMENTS")
println("="^50)

println("\n>>> Scenario 1: Stable Cartel (both f=0.8)")
st_cartel = run_stable_cartel(200)
analyze_results(st_cartel)

println("\n>>> Scenario 2: One Defects (f=1.0 vs f=0.8)")
st_defect = run_defection(200)
analyze_results(st_defect)

println("\n>>> Scenario 3: Full Competition (both f=1.0)")
st_compete = run_competition(200)
analyze_results(st_compete)

println("\n" * "="^50)
println("COMPARISON")
println("="^50)

# Profit comparison
cartel_avg = mean([sum(st_cartel.profits["Builder1"]), sum(st_cartel.profits["Builder2"])])
compete_avg = mean([sum(st_compete.profits["Builder1"]), sum(st_compete.profits["Builder2"])])
defect_winner = sum(st_defect.profits["Builder1"])
defect_loser = sum(st_defect.profits["Builder2"])

println("Average profit per builder:")
println("  Cartel:      ", round(cartel_avg, digits=2), " ETH")
println("  Competition: ", round(compete_avg, digits=2), " ETH")
println("\nDefection scenario:")
println("  Defector:    ", round(defect_winner, digits=2), " ETH")
println("  Cooperator:  ", round(defect_loser, digits=2), " ETH")

# Base fee comparison
println("\nAverage base fee:")
println("  Cartel:      ", round(mean(st_cartel.B_history), digits=2), " gwei")
println("  Competition: ", round(mean(st_compete.B_history), digits=2), " gwei")

# Stability check
if defect_winner > cartel_avg
    println("\n⚠️  CARTEL IS UNSTABLE: Defection is more profitable!")
else
    println("\n✓  CARTEL IS STABLE: Cooperation pays off")
end
