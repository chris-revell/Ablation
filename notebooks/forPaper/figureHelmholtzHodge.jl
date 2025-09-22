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

fig = Figure(size=(1500, 1000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

for (col,inputSystem) in enumerate(inputSystems)
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
    iⁱ = ones(Int64, I).-findPeripheralCells(B)
    jᵖ = findPeripheralEdges(B)
    jⁱⁿ = jᵖ.==0
    𝐜ⱼ = findEdgeMidpoints(R, A)

    cellAreas = findCellAreas(R, A, B)
    linkTriangles = findCellLinkTriangles(R, A, B)
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)
    # 𝐡 = [SVector{2,Float64}(1.0,0.0) for _=1:J]

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

    # 𝐯 = grad ϕ + rot u + x 
    # 𝐕 = grad ϕCapital + rot U + x 
    # grad ϕ = gradᵛ ϕpar + cogradᵛ ϕperp 
    # grad ϕCapital = gradᶜ ϕCapitalpar + cogradᶜ ϕCapitalperp 
    # rot u = rotᶜ uperp + corotᶜ upar 
    # rot U = rotᵛ Uperp + corotᵛ Upar

    𝐡_hh = gradᵛ(R, A, ϕpar2) + cogradᵛ(R, A, B, ϕperp2) + rotᶜ(R, A, B, uperp) + corotᶜ(R, A, B, upar) 
    𝐡_hh .= [𝐡_hh[j].-𝐡_hh[1] for j=1:size(B,2)]
    𝐇_hh = gradᶜ(R, A, B, ϕCapitalpar) + cogradᶜ(R, A, B, ϕCapitalperp) + rotᵛspokes(R, A, B, Uperp2) + corotᵛspokes(R, A, B, Upar2)
    𝐇_hh .= [𝐇_hh[j].-𝐇_hh[1] for j=1:size(B,2)]

    #%%
    gl = GridLayout(fig[1,col])
    
    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
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
    scatter!(axes[end], Point{2,Float64}.(𝐡[jⁱⁿ]), color=(:blue, 0.5))
    Label(gl[1,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
    
    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for i=1:I 
        poly!(axes[end], cellPolygons[i], color=(:white, 0.0), strokewidth=1, strokecolor=(:black,1.0))
    end
    # Label(fig[2, 3, Bottom()], "x")
    arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ)[jⁱⁿ], Point{2,Float64}.(([𝐡_hh[j].-𝐡[j] for j=1:size(B,2)])[jⁱⁿ]), color=:red, linewidth=2, lengthscale=20.0)
    Label(gl[2,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

    hidedecorations!(axes[end])
    hidespines!(axes[end])

end

colsize!(fig.layout, 1, Relative(0.33))
colsize!(fig.layout, 2, Relative(0.33))
colsize!(fig.layout, 3, Relative(0.33))

display(fig)
save(plotsdir(inputDir, "midpointsfigureHelmholtz.png"), fig)
