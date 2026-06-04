# ============================================================================
# File: OhMyU1.jl
# Part of: OhMyU1
#
# Description:
#   Module solves the constrained combinatorial optimization problem:
#   f(x) -> min; Ax = b; x \in Z.
#   The method of the solution is described here:
#   Alcazar, J., Ghazi Vakili, M., Kalayci, C.B. et al. Enhancing combinatorial 
#   optimization with classical and quantum generative models. Nat Commun 15, 
#   2761    (2024). https://doi.org/10.1038/s41467-024-46959-5
#
# Notes:
#   - Up to the now, it support binary vector x, integer matrix A and any cost
#     function f(x)
#   - Up to the now, it support only TT-ranks equal to 1
#   - Generic methods (for non-binary variables and r > 1) located in future.jl
# Author: Sergei
# Created: Jan 2026
# ============================================================================

module OhMyU1

using OMEinsum
using LinearAlgebra
using Random
using JuMP
using SCIP
using Base.Threads
using Profile
using Plots
using StaticArrays
using StatsBase
using StatsPlots
using ThreadsX
using SparseArrays

# include("py_tools.jl")
include("tools.jl")
include("mps_core.jl")
include("diversification.jl")
include("train.jl")
include("solver.jl")
# include("future.jl")

end 
