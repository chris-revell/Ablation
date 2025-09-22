using DrWatson
using DiscreteCalculus
using CairoMakie
using StaticArrays
using VertexModel
using LinearAlgebra
using SparseArrays
using Random
using Colors 
using JLD2
using Dates
using CircularArrays
using FromFile
using InvertedIndices

@from "$(srcdir("Stresses.jl"))" using Stresses

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

for inputSystem in inputSystems
    
    # Import system data
    fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
    importedData = load(fileName)
    R = importedData["R"]
    A = importedData["A"]
    B = importedData["B"]
    F = importedData["F"]

    I = size(B,1)
    J = size(B,2)
    K = size(A,2)
    aᵢ = findCellAreas(R, A, B)
    Eₖ = findCellLinkTriangleAreas(R, A, B)
    𝐜ⱼ = findEdgeMidpoints(R, A)
    𝐡 = hNetwork(R, A, B, F)
    
    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᵛh = curlᵛspokes(R, A, B, 𝐡)
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᵛh = divᵛsuppress(R, A, B, 𝐡)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    codivᵛh = codivᵛsuppress(R, A, B, 𝐡)


    

    @show inputSystem
    @show sum(aᵢ .* cocurlᶜh)
    @show sum(aᵢ .* curlᶜh)
    @show sum(-1.0.*aᵢ .* divᶜh)
    @show sum(-1.0.*aᵢ .* codivᶜh)

    println("")
    @show sum(Eₖ .* curlᵛh)
    
    println("")
    
    σᵢ = σ(R, A, B, 𝐡)
    pEffvals = pEff(R, A, B, 0.2, 0.75)
    
    printstyled("Validate cocurlᶜh==tr(σ)\n", color= (maximum(abs.(cocurlᶜh.-tr.(σᵢ)))<0.0000001 ? :green : :red))
    @show maximum(abs.(cocurlᶜh.-tr.(σᵢ)))
    #Validate cocurlᶜh==tr(σ)
    printstyled("Validate cocurlᶜh==2Peff\n", color= (maximum(abs.(cocurlᶜh.-2.0.*pEffvals))<0.0000001 ? :green : :red))
    @show maximum(abs.(cocurlᶜh.-2.0.*pEffvals))
    #Validate ∑aᵢtr(σᵢ)=0
    printstyled("Validate ∑aᵢtr(σᵢ)=0\n", color= (sum(aᵢ.*tr.(σᵢ))<0.0000001 ? :green : :red))
    @show sum(aᵢ.*tr.(σᵢ))
    @show sum(aᵢ.*0.5.*cocurlᶜh)

    println("")
end
