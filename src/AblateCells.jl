#
#  AblateCells.jl
#  Ablation 
#
#  Created by Christopher Revell on 14/05/2025.
#

module AblateCells

# Julia packages
using DiscreteCalculus
using SparseArrays

# Local modules
# @from "$(srcdir("X.jl"))" using X

function ablateCells(R, A, B, ablatedCellsList)
    newB = B[[ii for ii=1:size(B,1) if ii∉ablatedCellsList],[jj for jj=1:size(B,2)]]
    edgeCellPairs = [findall(x->x!=0,newB[:,j]) for j=1:size(newB,2)]
    orphanedEdges = findall(x->length(x)==0, edgeCellPairs)
    newB2 = newB[[ii for ii=1:size(newB,1)],[jj for jj=1:size(newB,2) if jj∉orphanedEdges]]
    newA = A[[jj for jj=1:size(A,1) if jj∉orphanedEdges],[kk for kk=1:size(A,2)]]
    edgeVertexPairs = [findall(x->x!=0,newA[:,k]) for k=1:size(newA,2)]
    orphanedVertices = findall(x->length(x)==0, edgeVertexPairs)
    newA2 = newA[[jj for jj=1:size(newA,1)],[kk for kk=1:size(newA,2) if kk∉orphanedVertices]]    
    senseCheck(newA2, newB2)
    newR = [R[k] for k=1:length(R) if k∉orphanedVertices]
    return newR, newA2, newB2
end 

function ablateEdge(A, B, j)
    j_is = first.(findnz(B[:,j]))
    newB = B[[ii for ii=1:size(B,1) if ii∉ablatedCellsList],[jj for jj=1:size(B,2)]]
    edgeCellPairs = first.([findnz(newB[:,j]) for j=1:size(newB,2)])
    orphanedEdges = findall(x->length(x)==0, edgeCellPairs)
    newB2 = newB[[ii for ii=1:size(newB,1)],[jj for jj=1:size(newB,2) if jj∉orphanedEdges]]
    newA = A[[jj for jj=1:size(A,1) if jj∉orphanedEdges],[kk for kk=1:size(A,2)]]
    senseCheck(newA, newB2)
    return newA, newB2
end 

export ablateCells
export ablateEdge

end #end module 
