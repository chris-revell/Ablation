#
#  File: PenrosePseudoInversion.jl
#  Module: PenrosePseudoInversion
#  Package: Ablation 
#  Functions: inProd, penrosePseudoInversion

module PenrosePseudoInversion

# Julia packages
using DiscreteCalculus
using SparseArrays
using LinearAlgebra

# Local modules
# @from "$(srcdir("X.jl"))" using X

inProd(a, M, b) = transpose(a) * M * b # Inner product of a and b under metric M

function penrosePseudoInversion(L, g, M)
    n = length(g)
    λᵛ = (eigen(Matrix(L))).values
    eᵛs = (eigen(Matrix(L))).vectors
    eᵛ = [eᵛs[:, k] for k = 1:n] # Vector of eigenvectors
    𝟙 = ones(n)
    ḡ = 𝟙 .* inProd(𝟙, M, g) / inProd(𝟙, M, 𝟙)
    ğ = g .- ḡ

    ϕ̆ = zeros(n)
    ϕ̆Spectrum = Float64[]
    ϕ̄₀ = zeros(n)
    for k = 2:n
        amplitude = inProd(eᵛ[k], M, ğ) / (λᵛ[k] * inProd(eᵛ[k], M, eᵛ[k]))
        ϕ̆ .+= amplitude .* eᵛ[k]
        push!(ϕ̆Spectrum, amplitude)

        # ϕ̄₀ .+= eᵛ[k] .* inProd(eᵛ[k], M, 𝟙) / (λᵛ[k] * inProd(eᵛ[k], M, eᵛ[k]))
    end

    # ϕ̄ = ϕ̄₀ * inProd(𝟙, M, g) / inProd(𝟙, M, 𝟙)
    ϕ = ϕ̆ #.+ ϕ̄

    return ϕ, ϕ̆Spectrum
end

export inProd
export penrosePseudoInversion

end
