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
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]
inputSystem = inputSystems[2]
fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
importedData = load(fileName)
R = importedData["R"]
A = importedData["A"]
B = importedData["B"]
F = importedData["F"]


# integ1 = vertexModel(abstol = 1e-9,
#                     reltol = 1e-9,
#                     nRows=7,
#                     nCycles=1,
#                     printToggle=1,
#                     frameDataToggle=0,
#                     frameImageToggle=0,
#                     videoToggle=0,
#                     setRandomSeed=12345,
#                     divisionToggle=0,
#                     pressureExternal=0.0,
#                     energyModel="quadratic",
#                 )
# R1 = reinterpret(SVector{2,Float64}, integ1.u) 
# params1, matrices1 = integ1.p
# A1 = matrices1.A
# B1 = matrices1.B
# integ2 = vertexModel(abstol = 1e-9,
#                     reltol = 1e-9,
#                     initialSystem="argument",
#                     divisionToggle=0,
#                     R_in=R1,
#                     A_in=A1,
#                     B_in=B1,
#                     pressureExternal=0.0,
#                     nCycles=1.0,
#                     outputToggle=0,
#                     frameDataToggle=0,
#                     frameImageToggle=0,
#                     videoToggle=0,
#                     printToggle=1,
#                     energyModel="quadratic",
#                 )

# R = reinterpret(SVector{2,Float64}, integ1.u) 
# params, matrices = integ1.p
# A = matrices.A
# B = matrices.B
# F = matrices.F

#%%

I = size(B,1)
J = size(B,2)
K = size(A,2)
iⁱ = ones(Int64, I).-findPeripheralCells(B)
jᵖ = findPeripheralEdges(B)
jⁱⁿ = jᵖ.==0
𝐜ⱼ = findEdgeMidpoints(R, A)

cellAreas = findCellAreas(R, A, B)
linkTriangles = findCellLinkTriangles(R, A, B)
linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
cellPolygons = findCellPolygons(R, A, B)
𝐡 = hNetwork(R, A, B, F)
# 𝐡 = 𝐜ⱼ

curlᶜh = curlᶜ(R, A, B, 𝐡)   
curlᵛh = curlᵛspokes(R, A, B, 𝐡)   
divᶜh = divᶜ(R, A, B, 𝐡)
divᵛh = divᵛsuppress(R, A, B, 𝐡)
cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)   
cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)  
codivᶜh = codivᶜ(R, A, B, 𝐡)
codivᵛh = codivᵛsuppress(R, A, B, 𝐡)

L̂v, Lvreindexing = geometricLvHatReduced(R, A, B)
L̂f, Lfreindexing = geometricLfHatReduced(R, A, B)
Lf = geometricLf(R, A, B)
L̂c, Lcreindexing = geometricLcHatReduced(R, A, B)
Lc = geometricLc(R, A, B)
L̂t, Ltreindexing = geometricLtHatReduced(R, A, B)
H = Diagonal(cellAreas[Lcreindexing])
E = Diagonal(linkTriangleAreas[Lvreindexing])

𝟙ᶜ = ones(I)
𝟙ᵛ = ones(K)

# ϕpar Lv -divᵛ
ϕpar = L̂v\(-1.0.*divᵛh[Lvreindexing])
ϕpar2 = zeros(K)
ϕpar2[Lvreindexing] .= ϕpar

# ϕperp Lv -codivᵛ
ϕperp = L̂v\(-1.0.*codivᵛh[Lvreindexing])
ϕperp2 = zeros(K)
ϕperp2[Lvreindexing] .= ϕperp
# upar Lf cocurlᶜ
upar = penrosePseudoInversion(L̂f, cocurlᶜh[Lfreindexing], H)
# uparCorrection = penrosePseudoInversion(Lf, ones(I), H)
# upar .+= uparCorrection .* innerProd(𝟙ᶜ, H, cocurlᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)

# uperp Lf curlᶜ
uperp = penrosePseudoInversion(L̂f, curlᶜh[Lfreindexing], H)
# uperpCorrection = penrosePseudoInversion(Lf, ones(I), H)
# uperp .+= uperpCorrection .* innerProd(𝟙ᶜ, H, curlᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
# ϕCapitalpar Lc -divᶜ
ϕCapitalpar = penrosePseudoInversion(L̂c, -1.0.*divᶜh[Lcreindexing], H)
ϕCapitalparCorrection = Lc\𝟙ᶜ
ϕCapitalpar .+= ϕCapitalparCorrection .* innerProd(𝟙ᶜ, H, -1.0.*divᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
# ϕCapitalperp Lc -codivᶜ
ϕCapitalperp = penrosePseudoInversion(L̂c, -1.0.*codivᶜh[Lcreindexing], H)
ϕCapitalperpCorrection = Lc\𝟙ᶜ
ϕCapitalperp .+= ϕCapitalperpCorrection .* innerProd(𝟙ᶜ, H, -1.0.*codivᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
# Upar Lt cocurlᵛ
Upar = L̂t\(cocurlᵛh[Ltreindexing])
Upar2 = zeros(K)
Upar2[Ltreindexing] .= Upar
# Uperp Lt curlᵛ
Uperp = L̂t\(curlᵛh[Ltreindexing])
Uperp2 = zeros(K)
Uperp2[Ltreindexing] .= Uperp

# 𝐯 = grad ϕ + rot u + x 
# 𝐕 = grad ϕCapital + rot U + x 
# grad ϕ = gradᵛ ϕpar + cogradᵛ ϕperp 
# grad ϕCapital = gradᶜ ϕCapitalpar + cogradᶜ ϕCapitalperp 
# rot u = rotᶜ uperp + corotᶜ upar 
# rot U = rotᵛ Uperp + corotᵛ Upar

kⁱ = findPeripheralVertices(A, B).==0
iⁱ = findPeripheralCells(B).==0

𝐡_hh = gradᵛ(R, A, ϕpar2) + cogradᵛ(R, A, B, ϕperp2) + rotᶜ(R, A, B, uperp) + corotᶜ(R, A, B, upar) 
𝐡_hh .= [𝐡_hh[j].-𝐡_hh[1] for j=1:J]
# @show maximum(abs.(divᵛh[kⁱ].-divᵛ(R, A, B, 𝐡_hh)[kⁱ]))
# @show maximum(abs.(codivᵛh[kⁱ].-codivᵛ(R, A, B, 𝐡_hh)[kⁱ]))
# @show maximum(abs.(cocurlᶜh[iⁱ].-cocurlᶜ(R, A, B, 𝐡_hh)[iⁱ]))
# @show maximum(abs.(curlᶜh[iⁱ].-curlᶜ(R, A, B, 𝐡_hh)[iⁱ]))

# @show maximum(abs.(divᵛ(R, A, B, gradᵛ(R, A, ϕpar2))[kⁱ])) #NZ
# @show maximum(abs.(divᵛ(R, A, B, cogradᵛ(R, A, B, ϕperp2))[kⁱ]))
# @show maximum(abs.(divᵛ(R, A, B, rotᶜ(R, A, B, uperp))[kⁱ]))
# @show maximum(abs.(divᵛ(R, A, B, corotᶜ(R, A, B, upar) )[kⁱ]))
# @show maximum(abs.(divᵛh[kⁱ].-divᵛ(R, A, B, gradᵛ(R, A, ϕpar2))[kⁱ]))

# @show maximum(abs.(codivᵛ(R, A, B, gradᵛ(R, A, ϕpar2))[kⁱ]))
# @show maximum(abs.(codivᵛ(R, A, B, cogradᵛ(R, A, B, ϕperp2))[kⁱ])) #NZ
# @show maximum(abs.(codivᵛ(R, A, B, rotᶜ(R, A, B, uperp))[kⁱ]))
# @show maximum(abs.(codivᵛ(R, A, B, corotᶜ(R, A, B, upar) )[kⁱ]))
# @show maximum(abs.(codivᵛh[kⁱ].-codivᵛ(R, A, B, cogradᵛ(R, A, B, ϕperp2))[kⁱ]))

# @show maximum(abs.(curlᶜ(R, A, B, gradᵛ(R, A, ϕpar2))[iⁱ]))
# @show maximum(abs.(curlᶜ(R, A, B, cogradᵛ(R, A, B, ϕperp2))[iⁱ]))
# @show maximum(abs.(curlᶜ(R, A, B, rotᶜ(R, A, B, uperp))[iⁱ])) #NZ but vv small
# @show maximum(abs.(curlᶜ(R, A, B, corotᶜ(R, A, B, upar) )[iⁱ]))
# @show maximum(abs.(curlᶜh[iⁱ].-curlᶜ(R, A, B, rotᶜ(R, A, B, uperp))[iⁱ]))

# @show maximum(abs.(cocurlᶜ(R, A, B, gradᵛ(R, A, ϕpar2))[iⁱ]))
# @show maximum(abs.(cocurlᶜ(R, A, B, cogradᵛ(R, A, B, ϕperp2))[iⁱ]))
# @show maximum(abs.(cocurlᶜ(R, A, B, rotᶜ(R, A, B, uperp))[iⁱ]))
# @show maximum(abs.(cocurlᶜ(R, A, B, corotᶜ(R, A, B, upar) )[iⁱ])) #NZ
# @show maximum(abs.(cocurlᶜh[iⁱ].-cocurlᶜ(R, A, B, corotᶜ(R, A, B, upar))[iⁱ])) #NZ
# # Issue is with corotᶜ

# #%%
# a = cocurlᶜh[iⁱ]
# b = cocurlᶜ(R, A, B, corotᶜ(R, A, B, upar))[iⁱ]
# c = a.-b
# alims = (-maximum(abs.(a)), maximum(abs.(a)))
# blims = (-maximum(abs.(b)), maximum(abs.(b)))
# clims = (-maximum(abs.(c)), maximum(abs.(c)))

# fig = Figure(size=(1500, 500), fontsize=24)
# ax = Axis(fig[1,1], aspect=DataAspect())
# for (i, ii) in enumerate(findall(x->x==1, iⁱ))
#     poly!(ax, cellPolygons[ii], color=a[i], colorrange=alims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,2], colorrange=alims, colormap=:bwr)
# ax1 = Axis(fig[1,3], aspect=DataAspect())
# for (i, ii) in enumerate(findall(x->x==1, iⁱ))
#     poly!(ax1, cellPolygons[ii], color=b[i], colorrange=blims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,4], colorrange=blims, colormap=:bwr)
# ax2 = Axis(fig[1,5], aspect=DataAspect())
# for (i, ii) in enumerate(findall(x->x==1, iⁱ))
#     poly!(ax2, cellPolygons[ii], color=c[i], colorrange=clims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,6], colorrange=clims, colormap=:bwr)
# display(fig)

#%%




𝐇_hh = gradᶜ(R, A, B, ϕCapitalpar) + cogradᶜ(R, A, B, ϕCapitalperp) + 2.0.*rotᵛspokes(R, A, B, Uperp2) + 2.0.*corotᵛspokes(R, A, B, Upar2)
𝐇_hh .= [𝐇_hh[j].-𝐇_hh[1] for j=1:J]
# @show maximum(abs.(divᶜh[iⁱ].-divᶜ(R, A, B, 𝐇_hh)[iⁱ]))
# @show maximum(abs.(codivᶜh[iⁱ].-codivᶜ(R, A, B, 𝐇_hh)[iⁱ]))
# @show maximum(abs.(cocurlᵛh[kⁱ].-cocurlᵛ(R, A, B, 𝐇_hh)[kⁱ]))
# @show maximum(abs.(curlᵛh[kⁱ].-curlᵛ(R, A, B, 𝐇_hh)[kⁱ]))

# @show maximum(abs.(divᶜ(R, A, B, gradᶜ(R, A, B, ϕCapitalpar))[iⁱ])) #NZ
# @show maximum(abs.(divᶜ(R, A, B, cogradᶜ(R, A, B, ϕCapitalperp))[iⁱ]))
# @show maximum(abs.(divᶜ(R, A, B, rotᵛ(R, A, B, Uperp2))[iⁱ]))
# @show maximum(abs.(divᶜ(R, A, B, corotᵛ(R, A, B, Upar2))[iⁱ]))
# @show maximum(abs.(divᶜh[iⁱ].-divᶜ(R, A, B, gradᶜ(R, A, B, ϕCapitalpar))[iⁱ])) 

# @show maximum(abs.(codivᶜ(R, A, B, gradᶜ(R, A, B, ϕCapitalpar))[iⁱ]))
# @show maximum(abs.(codivᶜ(R, A, B, cogradᶜ(R, A, B, ϕCapitalperp))[iⁱ])) #NZ
# @show maximum(abs.(codivᶜ(R, A, B, rotᵛ(R, A, B, Uperp2))[iⁱ]))
# @show maximum(abs.(codivᶜ(R, A, B, corotᵛ(R, A, B, Upar2))[iⁱ]))
# @show maximum(abs.(codivᶜh[iⁱ].-codivᶜ(R, A, B, cogradᶜ(R, A, B, ϕCapitalperp))[iⁱ]))

# @show maximum(abs.(cocurlᵛ(R, A, B, gradᶜ(R, A, B, ϕCapitalpar))[kⁱ]))
# @show maximum(abs.(cocurlᵛ(R, A, B, cogradᶜ(R, A, B, ϕCapitalperp))[kⁱ]))
# @show maximum(abs.(cocurlᵛ(R, A, B, rotᵛ(R, A, B, Uperp2))[kⁱ]))
# @show maximum(abs.(cocurlᵛ(R, A, B, corotᵛ(R, A, B, Upar2))[kⁱ])) #NZ 
# @show maximum(abs.(cocurlᵛh[kⁱ].-cocurlᵛ(R, A, B, corotᵛ(R, A, B, Upar2))[kⁱ]))

# @show maximum(abs.(curlᵛ(R, A, B, gradᶜ(R, A, B, ϕCapitalpar))[kⁱ]))
# @show maximum(abs.(curlᵛ(R, A, B, cogradᶜ(R, A, B, ϕCapitalperp))[kⁱ]))
# @show maximum(abs.(curlᵛ(R, A, B, rotᵛ(R, A, B, Uperp2))[kⁱ])) #NZ
# @show maximum(abs.(curlᵛ(R, A, B, corotᵛ(R, A, B, Upar2))[kⁱ]))
# @show maximum(abs.(curlᵛh[kⁱ].-curlᵛ(R, A, B, rotᵛ(R, A, B, Uperp2))[kⁱ]))



# #%%
# a = cocurlᵛh[kⁱ]
# alims = (-maximum(abs.(a)), maximum(abs.(a)))
# b = cocurlᵛ(R, A, B, corotᵛ(R, A, B, Upar2))[kⁱ]
# blims = (-maximum(abs.(b)), maximum(abs.(b)))
# c = a.-2.0.*b
# clims = (-maximum(abs.(c)), maximum(abs.(c)))

# fig = Figure(size=(1500, 500), fontsize=24)
# ax = Axis(fig[1,1], aspect=DataAspect())
# for (k, kk) in enumerate(findall(x->x==1, kⁱ))
#     poly!(ax, linkTriangles[kk], color=a[k], colorrange=alims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,2], colorrange=alims, colormap=:bwr)
# ax1 = Axis(fig[1,3], aspect=DataAspect())
# for (k, kk) in enumerate(findall(x->x==1, kⁱ))
#     poly!(ax1, linkTriangles[kk], color=b[k], colorrange=blims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,4], colorrange=blims, colormap=:bwr)
# ax2 = Axis(fig[1,5], aspect=DataAspect())
# for (k, kk) in enumerate(findall(x->x==1, kⁱ))
#     poly!(ax2, linkTriangles[kk], color=c[k], colorrange=clims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,6], colorrange=clims, colormap=:bwr)
# display(fig)

# #%%
# a = curlᵛh[kⁱ]
# alims = (-maximum(abs.(a)), maximum(abs.(a)))
# b = curlᵛ(R, A, B, rotᵛ(R, A, B, Uperp2))[kⁱ]
# blims = (-maximum(abs.(b)), maximum(abs.(b)))
# c = a.-2.0.*b
# clims = (-maximum(abs.(c)), maximum(abs.(c)))

# fig = Figure(size=(1500, 500), fontsize=24)
# ax = Axis(fig[1,1], aspect=DataAspect())
# for (k, kk) in enumerate(findall(x->x==1, kⁱ))
#     poly!(ax, linkTriangles[kk], color=a[k], colorrange=alims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,2], colorrange=alims, colormap=:bwr)
# ax1 = Axis(fig[1,3], aspect=DataAspect())
# for (k, kk) in enumerate(findall(x->x==1, kⁱ))
#     poly!(ax1, linkTriangles[kk], color=b[k], colorrange=blims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,4], colorrange=blims, colormap=:bwr)
# ax2 = Axis(fig[1,5], aspect=DataAspect())
# for (k, kk) in enumerate(findall(x->x==1, kⁱ))
#     poly!(ax2, linkTriangles[kk], color=c[k], colorrange=clims, colormap=:bwr, strokewidth=1, strokecolor=(:black,1.0))
# end
# Colorbar(fig[1,6], colorrange=clims, colormap=:bwr)
# display(fig)

#%%




fig = Figure(size=(1000, 1000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

push!(axes, Axis(fig[1,1], aspect=DataAspect()))
# for i=1:I
#     orderedVerts, orderedEdges = orderAroundCell(A, B, i)
#     lines!(axes[end], Point{2,Float64}.(𝐡[orderedEdges[0:end]]), color=(:blue, 0.5))
#     # lines!(axes[end], Point{2,Float64}.(𝐡_hh[orderedEdges[0:end]]), color=(:red, 0.5))
#     lines!(axes[end], Point{2,Float64}.(𝐇_hh[orderedEdges[0:end]]), color=(:green, 0.5))
# end
# for j=1:J
#     if jⁱⁿ[j] 
#         lines!(axes[end], Point{2,Float64}.([𝐡[j], 𝐡_hh[j]]), color=(:blue, 0.5))
#     end
# end
scatter!(axes[end], Point{2,Float64}.(𝐡_hh[jⁱⁿ]), color=(:red, 0.5))
# scatter!(axes[end], Point{2,Float64}.(𝐇_hh[jⁱⁿ]), color=(:green, 0.5))
# scatter!(axes[end], Point{2,Float64}.(𝐡[jⁱⁿ]), color=(:blue, 0.5))
Label(fig[2,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

push!(axes, Axis(fig[1,2], aspect=DataAspect()))
# for i=1:I
#     orderedVerts, orderedEdges = orderAroundCell(A, B, i)
#     lines!(axes[end], Point{2,Float64}.(𝐡[orderedEdges[0:end]]), color=(:blue, 0.5))
#     # lines!(axes[end], Point{2,Float64}.(𝐡_hh[orderedEdges[0:end]]), color=(:red, 0.5))
#     lines!(axes[end], Point{2,Float64}.(𝐇_hh[orderedEdges[0:end]]), color=(:green, 0.5))
# end
# for j=1:J
#     if jⁱⁿ[j] 
#         lines!(axes[end], Point{2,Float64}.([𝐡[j], 𝐡_hh[j]]), color=(:blue, 0.5))
#     end
# end
# scatter!(axes[end], Point{2,Float64}.(𝐡_hh[jⁱⁿ]), color=(:red, 0.5))
scatter!(axes[end], Point{2,Float64}.(𝐇_hh[jⁱⁿ]), color=(:green, 0.5))
# scatter!(axes[end], Point{2,Float64}.(𝐡[jⁱⁿ]), color=(:blue, 0.5))
Label(fig[2,2,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

push!(axes, Axis(fig[3,2], aspect=DataAspect()))
scatter!(axes[end], Point{2,Float64}.(𝐡[jⁱⁿ]), color=(:blue, 0.5))
Label(fig[4,2,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

push!(axes, Axis(fig[3,1], aspect=DataAspect()))
for i=1:I 
    poly!(axes[end], cellPolygons[i], color=(:white, 0.0), strokewidth=1, strokecolor=(:black,1.0))
end
# Label(fig[2, 3, Bottom()], "x")
𝐱 = [@SVector zeros(2) for j=1:J]
𝐱[jⁱⁿ] .= (𝐇_hh.-𝐡)[jⁱⁿ]
arrowColours = [(:red, norm(𝐱[j])/maximum(norm.(𝐱))) for j=1:J]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ)[jⁱⁿ], Point{2,Float64}.(𝐱[jⁱⁿ]), color=arrowColours[jⁱⁿ], linewidth=2, lengthscale=20.0)
Label(fig[4,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

hidedecorations!(axes[end])
hidespines!(axes[end])

colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.5))
rowsize!(fig.layout, 1, Relative(0.45))
rowsize!(fig.layout, 2, Relative(0.05))
rowsize!(fig.layout, 3, Relative(0.45))
rowsize!(fig.layout, 4, Relative(0.05))
display(fig)
# save(plotsdir(inputDir, "midpointsfigureHelmholtz.png"), fig)
