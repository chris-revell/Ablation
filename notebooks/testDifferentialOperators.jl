
using FromFile
using LinearAlgebra
using DrWatson
using VertexModel
using StaticArrays
using SparseArrays
using DiscreteCalculus

integ = vertexModel(nRows=11, nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0)
R = reinterpret(SVector{2,Float64}, integ.u) 
params, matrices = integ.p
@unpack A, B, F, cellAreas, peripheralVertices, peripheralEdges = matrices

ϕ = rand(size(A,2))
𝐛 = hNetwork(R, A, B, F)
f = cellAreas

dif = maximum(norm.(gradᵛ(R, A)*ϕ .-
    gradᵛ(R, A, ϕ)))
println("gradᵛ")
@show dif
dif = maximum(norm.(curlᶜ(R, A, B)*𝐛 .-
    curlᶜ(R, A, B, 𝐛)))
println("curlᶜ")
@show dif
dif = maximum(norm.(gradᶜ(R, A, B)*f .-
    gradᶜ(R, A, B, f)))
println("gradᶜ")
@show dif
dif = maximum(norm.(curlᵛ(R, A, B)*𝐛 .-
    curlᵛ(R, A, B, 𝐛))[peripheralVertices.==0])
println("curlᵛ")
@show dif
dif = maximum(norm.(rotᵛ(R, A, B)*ϕ .-
    rotᵛ(R, A, B, ϕ))[peripheralEdges.==0])
println("rotᵛ")
@show dif
dif = maximum(norm.(divᶜ(R, A, B)*𝐛 .-
    divᶜ(R, A, B, 𝐛)))
println("divᶜ")
@show dif
dif = maximum(norm.(rotᶜ(R, A, B)*f .-
    rotᶜ(R, A, B, f)))
println("rotᶜ")
@show dif
dif = maximum(norm.(divᵛ(R, A, B)*𝐛 .-
    divᵛ(R, A, B, 𝐛)))
println("divᵛ")
@show dif
dif = maximum(norm.(cogradᵛ(R, A, B)*ϕ .-
    cogradᵛ(R, A, B, ϕ)))
println("cogradᵛ")
@show dif
dif = maximum(norm.(cocurlᶜ(R, A, B)*𝐛 .-
    cocurlᶜ(R, A, B, 𝐛)))
println("cocurlᶜ")
@show dif
dif = maximum(norm.(cogradᶜ(R, A, B)*f .-
    cogradᶜ(R, A, B, f)))
println("cogradᶜ")
@show dif
dif = maximum(norm.(cocurlᵛ(R, A, B)*𝐛 .-
    cocurlᵛ(R, A, B, 𝐛))[peripheralVertices.==0])
println("cocurlᵛ")
@show dif
dif = maximum(norm.(corotᵛ(R, A, B)*ϕ .-
    corotᵛ(R, A, B, ϕ))[peripheralEdges.==0])
println("corotᵛ")
@show dif
dif = maximum(norm.(codivᶜ(R, A, B)*𝐛 .-
    codivᶜ(R, A, B, 𝐛)))
println("codivᶜ")
@show dif
dif = maximum(norm.(corotᶜ(R, A, B)*f .-
    corotᶜ(R, A, B, f)))
println("corotᶜ")
@show dif
dif = maximum(norm.(codivᵛ(R, A, B)*𝐛 .-
    codivᵛ(R, A, B, 𝐛)))
println("codivᵛ")
@show dif