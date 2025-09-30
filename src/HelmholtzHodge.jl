#
#  File: HelmholtzHodge.jl
#  Project: Ablation 
#

module HelmholtzHodge

# Julia packages
using DiscreteCalculus
using SparseArrays
using LinearAlgebra
using StaticArrays

function primalHH(R, A, B, ϕpar, ϕperp, upar, uperp)
    𝐯 = gradᵛ(R, A, ϕpar) .+ cogradᵛ(R, A, B, ϕperp) .+ rotᶜ(R, A, B, uperp) .+ corotᶜ(R, A, B, upar) 
    𝐯 .= [𝐯[j].-𝐯[1] for j=1:size(B,2)]
    return 𝐯
end

function dualHH(R, A, B, ϕCapitalpar, ϕCapitalperp, Upar, Uperp)
    𝐕 = gradᶜ(R, A, B, ϕCapitalpar) .+ cogradᶜ(R, A, B, ϕCapitalperp) .+ 2.0.*rotᵛ(R, A, B, Uperp) .+ 2.0.*corotᵛ(R, A, B, Upar)
    𝐕 .= [𝐕[j].-𝐕[1] for j=1:size(B,2)]
    return 𝐕
end

export primalHH
export dualHH

end #end module 
