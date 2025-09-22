#
#  Stresses.jl
#  Ablation 
#
#  Created by Christopher Revell on 08/08/2025.
#

module Stresses

# Julia packages
using DiscreteCalculus
using SparseArrays
using LinearAlgebra
using StaticArrays

# ~\ref{eq:shearstress}
function ζ(R, A, B, m, zperp)
    # Lprimal = edgeLaplacianPrimal(R, A, B)
    # wᵐ = (eigen(Matrix(Lprimal)).vectors)[:,m]
    # 𝐭̂ = normalize.(findEdgeTangents(R, A))
    # aᵢ = findCellAreas(R, A, B)
    # tmp = [B[i,j].*wᵐ[j].*outerProd(𝐭̂[j],𝐭̂[j])./aᵢ[i] for i=1:size(B,1), j=1:size(B,2)]
    # ζᵐᵢtmp = dropdims(sum(tmp, dims=2), dims=2)
    # ζᵐᵢ = zperp.*sqrt.(-1.0.*det.(ζᵐᵢtmp))
    # return ζᵐᵢ
    B̂ = findB̂(A, B)
    jᵖ = findPeripheralEdges(B)
    Lprimal = edgeLaplacianPrimalHat(R, A, B)
    wᵐ = (eigen(Matrix(Lprimal)).vectors)[:,m]
    𝐭̂ = normalize.(findEdgeTangents(R, A))[jᵖ.==0]
    aᵢ = findCellAreas(R, A, B)
    tmp = [B̂[i,j].*wᵐ[j].*outerProd(𝐭̂[j],𝐭̂[j])./aᵢ[i] for i=1:size(B,1), j=1:(size(B,2)-sum(jᵖ))]
    ζᵐᵢtmp = dropdims(sum(tmp, dims=2), dims=2)
    ζᵐᵢ = zperp.*sqrt.(-1.0.*det.(ζᵐᵢtmp))
    return ζᵐᵢ
end 


function σ(R, A, B, 𝐡ⱼ)
    I = size(B,1)
    J = size(B,2)
    aᵢ = findCellAreas(R, A, B)
    ϵᵢ = SMatrix{2, 2, Float64}([
        0.0 1.0
        -1.0 0.0
    ])
    𝐭ⱼ = findEdgeTangents(R, A)
    tmp = [B[i,j]*outerProd(𝐭ⱼ[j], 𝐡ⱼ[j])*ϵᵢ./aᵢ[i] for i=1:I, j=1:J]
    return dropdims(sum(tmp, dims=2), dims=2)
end

pEff(cellAreas, cellPressures, cellPerimeters, cellTensions) = cellPressures .- cellTensions.*cellPerimeters./(2.0.*cellAreas)
function pEff(R, A, B, γ, L₀; forceModel="log") 
    cellAreas = findCellAreas(R, A, B)
    cellPerimeters = findCellPerimeterLengths(R, A, B)
    if forceModel=="log"
        cellTensions = γ .* L₀ .* log.(cellPerimeters ./ L₀)
        cellPressures = log.(cellAreas)
    else
        cellTensions = γ .*(cellPerimeters .- L₀)
        cellPressures = cellAreas
    end
    pEff = cellPressures .- cellTensions.*cellPerimeters./(2.0.*cellAreas)
    return pEff
end

export ζ
export σ
export pEff

end #end module 
