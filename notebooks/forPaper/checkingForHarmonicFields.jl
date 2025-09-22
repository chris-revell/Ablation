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

@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir)); isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]

#%%
    
inputSystem = inputSystems[1]
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

Â = findÂ(A,B)
Aᵀ = Transpose(A)
Âᵀ = Transpose(Â)
B̂ = findB̂(A,B)
Bᵀ = Transpose(B)
B̂ᵀ = Transpose(B̂)

# println("AᵀA:")
λAᵀA1 = eigen(Matrix(Aᵀ*A)).values
wAᵀA1 = [eigen(Matrix(Aᵀ*A)).vectors[:,k] for k=1:size(A,2)]
λÂᵀÂ1 = eigen(Matrix(Âᵀ*Â)).values
wÂᵀÂ1 = [eigen(Matrix(Âᵀ*Â)).vectors[:,k] for k=1:size(Â,2)]
λBBᵀ1 = eigen(Matrix(B*Bᵀ)).values
wBBᵀ1 = [eigen(Matrix(B*Bᵀ)).vectors[:,i] for i=1:size(B,1)]
λB̂B̂ᵀ1 = eigen(Matrix(B̂*B̂ᵀ)).values
wB̂B̂ᵀ1 = [eigen(Matrix(B̂*B̂ᵀ)).vectors[:,i] for i=1:size(B̂,1)]

# @show λAᵀA1[1:6]
# @show wAᵀA1[1:6]
# @show λÂᵀÂ1[1:6]
# @show wÂᵀÂ1[1:6]
# @show λBBᵀ1[1:6]
# @show wBBᵀ1[1:6]
# @show λB̂B̂ᵀ1[1:6]
# @show wB̂B̂ᵀ1[1:6]


#%%
    
inputSystem = inputSystems[2]
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

Â = findÂ(A,B)
Aᵀ = Transpose(A)
Âᵀ = Transpose(Â)
B̂ = findB̂(A,B)
Bᵀ = Transpose(B)
B̂ᵀ = Transpose(B̂)

# println("AᵀA:")
λAᵀA2 = eigen(Matrix(Aᵀ*A)).values
wAᵀA2 = [eigen(Matrix(Aᵀ*A)).vectors[:,k] for k=1:size(A,2)]
λÂᵀÂ2 = eigen(Matrix(Âᵀ*Â)).values
wÂᵀÂ2 = [eigen(Matrix(Âᵀ*Â)).vectors[:,k] for k=1:size(Â,2)]
λBBᵀ2 = eigen(Matrix(B*Bᵀ)).values
wBBᵀ2 = [eigen(Matrix(B*Bᵀ)).vectors[:,i] for i=1:size(B,1)]
λB̂B̂ᵀ2 = eigen(Matrix(B̂*B̂ᵀ)).values
wB̂B̂ᵀ2 = [eigen(Matrix(B̂*B̂ᵀ)).vectors[:,i] for i=1:size(B̂,1)]

# @show λAᵀA2[1:6]
# @show wAᵀA2[1:6]
# @show λÂᵀÂ2[1:6]
# @show wÂᵀÂ2[1:6]
# @show λBBᵀ2[1:6]
# @show wBBᵀ2[1:6]
# @show λB̂B̂ᵀ2[1:6]
# @show wB̂B̂ᵀ2[1:6]


#%%
    
inputSystem = inputSystems[3]
# Import system data
fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
importedData = load(fileName)
R = importedData["R"]
A = importedData["A"]
B = importedData["B"]
F = importedData["F"]

Â = findÂ(A,B)
Aᵀ = Transpose(A)
Âᵀ = Transpose(Â)
B̂ = findB̂(A,B)
Bᵀ = Transpose(B)
B̂ᵀ = Transpose(B̂)

# println("AᵀA:")
λAᵀA3 = eigen(Matrix(Aᵀ*A)).values
wAᵀA3 = [eigen(Matrix(Aᵀ*A)).vectors[:,k] for k=1:size(A,2)]
λÂᵀÂ3 = eigen(Matrix(Âᵀ*Â)).values
wÂᵀÂ3 = [eigen(Matrix(Âᵀ*Â)).vectors[:,k] for k=1:size(Â,2)]
λBBᵀ3 = eigen(Matrix(B*Bᵀ)).values
wBBᵀ3 = [eigen(Matrix(B*Bᵀ)).vectors[:,i] for i=1:size(B,1)]
λB̂B̂ᵀ3 = eigen(Matrix(B̂*B̂ᵀ)).values
wB̂B̂ᵀ3 = [eigen(Matrix(B̂*B̂ᵀ)).vectors[:,i] for i=1:size(B̂,1)]

# @show λAᵀA3[1:6]
# @show wAᵀA3[1:6]
# @show λÂᵀÂ3[1:6]
# @show wÂᵀÂ3[1:6]
# @show λBBᵀ3[1:6]
# @show wBBᵀ3[1:6]
# @show λB̂B̂ᵀ3[1:6]
# @show wB̂B̂ᵀ3[1:6]
