#
#  File: PenrosePseudoInversion.jl
#  Module: PenrosePseudoInversion
#  Package: Ablation 
#  Functions: innerProd, penrosePseudoInversion

module PenrosePseudoInversion

# Julia packages
using DiscreteCalculus
using SparseArrays
using LinearAlgebra

# Local modules
# @from "$(srcdir("X.jl"))" using X

function penrosePseudoInversion(L, g, M)
    n = length(g)
    λᵛ = (eigen(Matrix(L))).values
    eᵛs = (eigen(Matrix(L))).vectors
    eᵛ = [eᵛs[:, k] for k = 1:n] # Vector of eigenvectors
    𝟙 = ones(n)
    ḡ = 𝟙 .* innerProd(𝟙, M, g) / innerProd(𝟙, M, 𝟙)
    ğ = g .- ḡ

    ϕ̆ = zeros(n)
    # ϕ̆Spectrum = Float64[]
    for k = 2:n
    # for k = findfirst(x->x>1e-9, λᵛ):n
        amplitude = innerProd(eᵛ[k], M, ğ) / (λᵛ[k] * innerProd(eᵛ[k], M, eᵛ[k]))
        ϕ̆ .+= amplitude .* eᵛ[k]
        # push!(ϕ̆Spectrum, amplitude)
        # ϕ̄₀ .+= eᵛ[k] .* innerProd(eᵛ[k], M, 𝟙) / (λᵛ[k] * innerProd(eᵛ[k], M, eᵛ[k]))
    end

    # ϕ̄ = ϕ̄₀ * innerProd(𝟙, M, g) / innerProd(𝟙, M, 𝟙)
    # ϕ = ϕ̆ .+ ϕ̄

    return ϕ̆ #, ϕ̆Spectrum
end

export innerProd
export penrosePseudoInversion

end
