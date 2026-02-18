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
using FromFile
using InvertedIndices

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses
@from "$(srcdir("HelmholtzHodge.jl"))" using HelmholtzHodge
@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion

inputSystem = "Large2"
inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

inFile = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData

systemCOM = sum(R)./length(R)
cellCentres1 = findCellCentresOfMass(R, A, B)
centralCell = findmin(norm.([cellCentres1[i].-systemCOM for i=1:size(B,1)]))[2]

neighbourMatrix = dropzeros(B*transpose(B))
ablatedCells = [centralCell]

if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"))
    inFile = datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2")
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
    #%%
    params2, matrices2 = integ2.p
    @show maximum(norm.(sum(F2, dims=2)))
    R2 = reinterpret(SVector{2,Float64}, integ2.u)
    A2 = matrices2.A
    B2 = matrices2.B
    F2 = matrices2.F

    C = findC(A, B)
    centralCellVertices = R[findall(x->x!=0, C[centralCell, :])]
    systemCOM2 = sum(centralCellVertices)./length(centralCellVertices)
    
    jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"); 
        R2,
        A2, 
        B2, 
        F2, 
        systemCOM2,
    )
end

I = size(B2,1)
J = size(B2,2)
K = size(A2,2)
iⁱ = ones(Int64, I).-findPeripheralCells(B2)
jᵖ = findPeripheralEdges(B2)
jⁱⁿ = jᵖ.==0
𝐜ⱼ2 = findEdgeMidpoints(R2, A2)
𝐂ⱼ2 = findCellLinkMidpoints(R2, A2, B2)

cellAreas2 = findCellAreas(R2, A2, B2)
linkTriangles2 = findCellLinkTriangles(R2, A2, B2)
linkTriangleAreas2 = findCellLinkTriangleAreas(R2, A2, B2)
cellPolygons2 = findCellPolygons(R2, A2, B2)


𝐡 = hNetwork(R2, A2, B2, F2)

curlᶜh = curlᶜ(R2, A2, B2, 𝐡)   
curlᵛh = curlᵛspokes(R2, A2, B2, 𝐡)   
divᶜh = divᶜ(R2, A2, B2, 𝐡)
divᵛh = divᵛsuppress(R2, A2, B2, 𝐡)
cocurlᶜh = cocurlᶜ(R2, A2, B2, 𝐡)   
cocurlᵛh = cocurlᵛspokes(R2, A2, B2, 𝐡)  
codivᶜh = codivᶜ(R2, A2, B2, 𝐡)
codivᵛh = codivᵛsuppress(R2, A2, B2, 𝐡)

L̂v, Lvreindexing = geometricLvHatReduced(R2, A2, B2)
L̂f, Lfreindexing = geometricLfHatReduced(R2, A2, B2)
Lf = geometricLf(R2, A2, B2)
L̂c, Lcreindexing = geometricLcHatReduced(R2, A2, B2)
Lc = geometricLc(R2, A2, B2)
L̂t, Ltreindexing = geometricLtHatReduced(R2, A2, B2)
H = Diagonal(cellAreas2[Lcreindexing])
E = Diagonal(linkTriangleAreas2[Lvreindexing])

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
uparCorrection = penrosePseudoInversion(Lf, ones(I), H)
upar .+= uparCorrection .* innerProd(𝟙ᶜ, H, cocurlᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
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




# 𝐡_hh = primalHH(R2, A2, B2, ϕpar2, ϕperp2, upar, uperp)
# 𝐇_hh = dualHH(R2, A2, B2, ϕCapitalpar, ϕCapitalperp, Upar2, Uperp2)
# 𝐱 = 𝐡.-𝐡_hh
# 𝐗 = 𝐡.-𝐇_hh

# 𝐡_hh = gradᵛ(R2, A2, ϕpar2) .+ cogradᵛ(R2, A2, B2, ϕperp2) .+ rotᶜ(R2, A2, B2, uperp) .+ corotᶜ(R2, A2, B2, upar) 
# 𝐡_hh .= [𝐡_hh[j].-𝐡_hh[1] for j=1:size(B2,2)]
𝐡_hh = primalHH(R2, A2, B2, ϕpar2, ϕperp2, upar, uperp)
𝐱 = 𝐡.-𝐡_hh
𝐇_hh = dualHH(R2, A2, B2, ϕCapitalpar, ϕCapitalperp, Upar2, Uperp2)
𝐗 = 𝐡.-𝐇_hh

edgeradii2 = norm.([𝐜ⱼ2[j].-systemCOM2 for j=1:size(A2,1)])
linkradii2 = norm.([𝐂ⱼ2[j].-systemCOM2 for j=1:size(A2,1)])
edgeDummyDists2 = collect(maximum(edgeradii2)/100:maximum(edgeradii2)/100:maximum(edgeradii2))
linkDummyDists2 = collect(maximum(linkradii2)/100:maximum(linkradii2)/100:maximum(linkradii2))

#%%

fig = Figure(size=(1000,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

push!(axes, Axis(fig[1,1], aspect=DataAspect()))
for i=1:I 
    poly!(axes[end], cellPolygons2[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
end
arrowColours = [(:blue, norm(𝐱[j])/maximum(norm.(𝐱[jⁱⁿ]))) for j=1:J]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ2)[jⁱⁿ], Vec{2,Float64}.(𝐱[jⁱⁿ]), color=arrowColours[jⁱⁿ], linewidth=2, lengthscale=20.0)
Label(fig[2,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
hidedecorations!(axes[end])
hidespines!(axes[end])

push!(axes, Axis(fig[1,2], aspect=AxisAspect(1.25)))
scatter!(axes[end], log10.(edgeradii2[jⁱⁿ]), log10.(norm.(𝐱[jⁱⁿ])), color=(:blue,0.3))
lines!(axes[end], log10.(edgeDummyDists2), log10.(0.02./edgeDummyDists2), color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], log10.(edgeDummyDists2), log10.(0.02./edgeDummyDists2.^2), color=(:black, 0.75), linestyle=:dash)
axes[end].xlabel = L"\log_{10}\left(c_j\right)"
axes[end].ylabel = L"\log_{10}\left(|\breve{\mathbf{x}}_j|\right)"
xlims!(axes[end], (-0.6, 1.0))
ylims!(axes[end], (-3.6, -1.0))
Label(fig[2,2,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

# push!(axes, Axis(fig[3,1], aspect=DataAspect()))
# for i=1:I 
#     poly!(axes[end], cellPolygons2[i], color=(:white, 0.0), strokewidth=1, strokecolor=(:black,0.1))
# end
# arrowColours = [(:green, norm(𝐗[j])/maximum(norm.(𝐗[jⁱⁿ]))) for j=1:J]
# arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ2)[jⁱⁿ], Vec{2,Float64}.(𝐗[jⁱⁿ]), color=arrowColours[jⁱⁿ], linewidth=2, lengthscale=20.0)
# Label(fig[4,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
# hidedecorations!(axes[end])
# hidespines!(axes[end])

# push!(axes, Axis(fig[3,2], aspect=AxisAspect(1.25)))
# scatter!(axes[end], log10.(linkradii2[jⁱⁿ]), log10.(norm.(𝐗[jⁱⁿ])), color=(:green,0.3))
# lines!(axes[end], log10.(linkDummyDists2), log10.(0.02./linkDummyDists2), color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], log10.(linkDummyDists2), log10.(0.02./linkDummyDists2.^2), color=(:black, 0.75), linestyle=:dash)
# axes[end].xlabel = L"\log_{10}\left(r_j\right)"
# axes[end].ylabel = L"\log_{10}\left(|X|\right)"
# xlims!(axes[end], (-0.6, 1.0))
# ylims!(axes[end], (-3.6, -1.0))
# Label(fig[4,2,Bottom()], popfirst!(subfigureLabels), fontsize=24) 


rowsize!(fig.layout, 1, Relative(0.98))
rowsize!(fig.layout, 2, Relative(0.02))
# rowsize!(fig.layout, 3, Relative(0.48))
# rowsize!(fig.layout, 4, Relative(0.01))
colsize!(fig.layout, 1, Relative(0.45))
colsize!(fig.layout, 2, Relative(0.55))

display(fig)

save(plotsdir(inputDir, "Figure10xFieldCellRemoved.png"), fig)
save(plotsdir(inputDir, "Figure10xFieldCellRemoved.pdf"), fig)


