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

@from "$(srcdir("AblateCells.jl"))" using AblateCells

# ablatedVersion = "LargerHole"
ablatedVersion = ""
inputSystem = "Large2"
inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

inFile = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData

systemCOM = sum(R)./length(R)
cellCentres1 = findCellCentresOfMass(R, A, B)
centralCell = findmin(norm.([cellCentres1[i].-systemCOM for i=1:size(B,1)]))[2]

if ablatedVersion=="LargerHole"
    ablatedCells = [centralCell]
    neighbourMatrix = dropzeros(B*transpose(B))
    ablatedCells = unique(findall(x->x!=0, neighbourMatrix[centralCell,:]))
    for _ = 1:2
        neighbours = getindex.(findall(x->x!=0, neighbourMatrix[ablatedCells,:]),2)
        append!(ablatedCells, neighbours)
    end
    unique!(ablatedCells)
else
    ablatedCells = [centralCell]
end

if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)$(ablatedVersion)Ablated_testSystem.jld2"))
    inFile = datadir("referenceSystems", inputDir, "$(inputSystem)$(ablatedVersion)Ablated_testSystem.jld2")
    importedData = load(inFile)
    @unpack R2, A2, B2, F2, systemCOM2 = importedData
else
    Rtmp, Atmp, Btmp = ablateCells(R, A, B, ablatedCells)

    integ2 = vertexModel(abstol = 1e-9,
                        reltol = 1e-9,
                        initialSystem="argument",
                        divisionToggle=0,
                        R_in=Rtmp,
                        A_in=Atmp,
                        B_in=Btmp,
                        pressureExternal=0.0,
                        nCycles=0.5,
                        outputToggle=0,
                        frameDataToggle=0,
                        frameImageToggle=0,
                        videoToggle=0,
                        printToggle=1,
                        energyModel="quadratic",
                    )

    params2, matrices2 = integ2.p
    
    R2 = reinterpret(SVector{2,Float64}, integ2.u)
    A2 = matrices2.A
    B2 = matrices2.B
    F2 = matrices2.F
    @show maximum(norm.(sum(F2, dims=2)))

    C = findC(A, B)
    centralCellVertices = R[findall(x->x!=0, C[centralCell, :])]
    systemCOM2 = sum(centralCellVertices)./length(centralCellVertices)
    
    jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)$(ablatedVersion)Ablated_testSystem.jld2"); 
        R2,
        A2, 
        B2, 
        F2, 
        systemCOM2,
    )
end

#%%

I = size(B2,1)
J = size(B2,2)
K = size(A2,2)
iⁱ = ones(Int64, I).-findPeripheralCells(B2)
jᵖ = findPeripheralEdges(B2)
jⁱⁿ = jᵖ.==0
𝐜ⱼ2 = findEdgeMidpoints(R2, A2)
𝐂ⱼ2 = findCellLinkMidpoints(R2, A2, B2)
𝐬ᵢₖ = findEdgeMidpointLinks(R2, A2, B2)
𝐄ₖ = findCellLinkVertexTriangles(R2, A2, B2)

cellCentres2 = findCellCentresOfMass(R2, A2, B2)


# Lprimal = edgeLaplacianPrimalHat(R, A, B)
Lprimal = edgeLaplacianPrimal(R2, A2, B2)
# Ldual = edgeLaplacianDualHat(R2, A2, B2)
Ldual = edgeLaplacianDual(R2, A2, B2)
μLprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
λLprimal = (eigen(Matrix(Lprimal))).values
μLdual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
λLdual = (eigen(Matrix(Ldual))).values
cellPolygons = findCellPolygons(R2, A2, B2)
# 𝐞Parallel = findEdgeTangents(R2, A2)[jⁱ.==1]./(findEdgeLengths(R2, A2)[jⁱ.==1].^2)
𝐞Parallel = findEdgeTangents(R2, A2)./(findEdgeLengths(R2, A2).^2)
𝐞Perp = [ϵᵢ*v for v in 𝐞Parallel]
# 𝐄Parallel = findCellLinks(R2, A2, B2)[jⁱ.==1]./(findCellLinkLengths(R2, A2, B2)[jⁱ.==1].^2)
𝐄Parallel = findCellLinks(R2, A2, B2)./(findCellLinkLengths(R2, A2, B2).^2)
𝐄Perp = [ϵₖ*v for v in 𝐄Parallel]

(zpar, zperp) = (1,0)
# Primal network 
harmonicFieldEdges = μLprimal[1].*(zpar.*𝐞Parallel .+ zperp.*𝐞Perp)
# Dual network 
harmonicFieldLinks = μLdual[1].*(zpar.*𝐄Parallel .+ zperp.*𝐄Perp)



𝐃c𝐯 = 𝐃c(R2, A2, B2, harmonicFieldEdges)
𝐆v𝐯 = 𝐆v(R2, A2, B2, 𝐃c𝐯)

cocurlᶜ𝐯 = cocurlᶜ(R2, A2, B2, harmonicFieldEdges)
Trace𝐃c𝐯 = tr.(𝐃c𝐯)
@show maximum(abs.(cocurlᶜ𝐯.-Trace𝐃c𝐯))
curlᶜ𝐯 = curlᶜ(R2, A2, B2, harmonicFieldEdges)
Antisym𝐃c𝐯 = 0.5.*(𝐃c𝐯.-Transpose.(𝐃c𝐯))
@show maximum(abs.(curlᶜ𝐯.-getindex.(Antisym𝐃c𝐯,2)))

u = normalize([1.0,0.0])
uPerp = normalize([0.0,1.0])
testArrows𝐃c𝐯 = [𝐃c𝐯[i]*u for i=1:size(B2,1)]
testVals𝐆v𝐯 = [(𝐆v𝐯[i]*u)[1] for i=1:size(A2,2)]

𝐧ᵢⱼ = findCellOutwardNormals(R2, A2, B2)
aᵢ = findCellAreas(R, A, B)
Dₖ = findEdgeMidpointLinkVertexAreas(R, A, B)

# tmp = [(uPerp⋅(ϵᵢ*𝐬ᵢₖ[i,k])*(u⋅𝐧ᵢⱼ[i,j])/(aᵢ[i]*Dₖ[k]))*harmonicFieldEdges[j] for i=1:size(B2,1), j=1:size(B2,2), k=1:size(A2,2)]

testVals𝐆v𝐯2 = fill(SVector{2,Float64}(zeros(2)), size(A2,2))
is, ks, vals = findnz(𝐬ᵢₖ)
for ind = 1:length(ks)
    k = ks[ind]
    i = is[ind]
    i_js = findall(x->x!=0, 𝐧ᵢⱼ[i,:])
    for j in i_js
        testVals𝐆v𝐯2[k] -= (uPerp⋅(ϵᵢ*𝐬ᵢₖ[i,k])*(u⋅𝐧ᵢⱼ[i,j])/(aᵢ[i]*Dₖ[k]))*harmonicFieldEdges[j]
    end
end



#%%


fig = Figure(size=(1500,1000), fontsize=24)

ax1a = Axis(fig[1,1], aspect=DataAspect())
hidedecorations!(ax1a); hidespines!(ax1a)
cellPolygons = findCellPolygons(R2, A2, B2)
for i=1:size(B2,1)
    poly!(ax1a, cellPolygons[i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end
arrows2d!(ax1a, 
    Point{2,Float64}.(𝐜ⱼ2), 
    Vec{2,Float64}.(harmonicFieldEdges),
    color=:red,
    lengthscale=1.0,
    )
Label(fig[1,1,Bottom()], L"\breve{x}_j")

ax1b = Axis(fig[2,1])
scatter!(ax1b, log.(norm.([𝐜ⱼ2[i].-systemCOM2 for i=1:size(B2,2)])), log.(norm.(harmonicFieldEdges)), color=(:blue, 0.2))
xs = collect(-2.0:0.1:2.0)
lines!(ax1b, xs, -1.0.*xs.-1.5, color=:red, linewidth=2)
lines!(ax1b, xs, -2.0.*xs.-1.5, color=:red, linewidth=2)
lines!(ax1b, xs, -3.0.*xs.-1.5, color=:red, linewidth=2)
ax1b.xlabel = L"r"
ax1b.ylabel = L"\log (|\mathbf{x}|)"


ax2a = Axis(fig[1,2], aspect=DataAspect())
hidedecorations!(ax2a); hidespines!(ax2a)
for i=1:size(B2,1)
    poly!(ax2a, cellPolygons[i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end
arrows2d!(ax2a, 
    Point{2,Float64}.(cellCentres2), 
    Vec{2,Float64}.(testArrows𝐃c𝐯),
    color=:red,
    lengthscale=1.0,
    )
Label(fig[1,2, Bottom()], L"\mathbf{u}\cdot \mathbf{D}_c \mathbf{\breve{x}}")

ax2b = Axis(fig[2,2])
scatter!(ax2b, log.(norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])), log.(norm.(testArrows𝐃c𝐯)), color=(:blue, 0.2))
xs = collect(-1.0:0.1:2.0)
lines!(ax2b, xs, -1.0.*xs.-2.0, color=:red, linewidth=2)
lines!(ax2b, xs, -2.0.*xs.-2.0, color=:red, linewidth=2)
lines!(ax2b, xs, -3.0.*xs.-2.0, color=:red, linewidth=2)
ax2b.xlabel = L"r"
ax2b.ylabel = L"\log (|\mathbf{u}\cdot \mathbf{D}_c \mathbf{x}|)"


ax3a = Axis(fig[1,3], aspect=DataAspect())
hidedecorations!(ax3a); hidespines!(ax3a)
for i=1:size(B2,1)
    poly!(ax3a, cellPolygons[i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end

radialVectors = normalize.([R2[k].-systemCOM2 for k=1:size(A2,2)])
radialComponents = (normalize.(testVals𝐆v𝐯2).⋅radialVectors)
# arrows2d!(ax3a, 
#     Point{2,Float64}.(R2), 
#     Vec{2,Float64}.(radialComponents),
#     # Vec{2,Float64}.(testVals𝐆v𝐯2),
#     color=:red,
#     lengthscale=0.01,
#     )
clims = (-maximum(abs.(radialComponents)), maximum(abs.(radialComponents)))
for k=1:size(A2,2)
    poly!(ax3a, 𝐄ₖ[k], color=radialComponents[k], colormap = :bwr, colorrange = clims, strokecolor=(:black, 0.5), strokewidth=1)
end
Label(fig[1,3, Bottom()], L"\mathbf{u}\cdot \mathbf{G}_c (\mathbf{D}_c \mathbf{\breve{x}})")

ax3b = Axis(fig[2,3])
# scatter!(ax3b, log.(norm.([R2[i].-systemCOM2 for i=1:size(A2,2)])), log.(norm.(testVals𝐆v𝐯)), color=(:blue, 0.2))
scatter!(ax3b, log.(norm.([R2[i].-systemCOM2 for i=1:size(A2,2)])), log.(norm.(radialComponents)), color=(:blue, 0.2))
xs = collect(-2.0:0.1:2.0)
lines!(ax3b, xs, -1.0.*xs, color=:red, linewidth=2)
lines!(ax3b, xs, -2.0.*xs, color=:red, linewidth=2)
lines!(ax3b, xs, -3.0.*xs, color=:red, linewidth=2)
ax3b.xlabel = L"r"
ax3b.ylabel = L"\log (|\mathbf{u}\cdot \mathbf{G}_c (\mathbf{D}_c \mathbf{\breve{x}})|)"



colsize!(fig.layout, 1, Relative(0.33))
colsize!(fig.layout, 2, Relative(0.33))
colsize!(fig.layout, 3, Relative(0.33))
rowsize!(fig.layout, 1, Relative(0.6))
rowsize!(fig.layout, 2, Relative(0.4))

display(fig)

#%%


# 𝐃v𝐕 = 𝐃v(R, A, B, findCellLinkMidpoints(R, A, B))
# cocurlᵛ𝐕 = cocurlᵛ(R, A, B, harmonicFieldLinks)
# Trace𝐃v𝐕 = tr.(𝐃v𝐕)
# @show maximum(abs.(cocurlᵛ𝐕.-Trace𝐃v𝐕))
