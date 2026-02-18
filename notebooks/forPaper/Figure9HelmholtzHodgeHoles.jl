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
    𝐂ⱼ = findCellLinkMidpoints(R, A, B)

    cellAreas = findCellAreas(R, A, B)
    linkTriangles = findCellLinkTriangles(R, A, B)
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)

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

    𝐡_hh = primalHH(R, A, B, ϕpar2, ϕperp2, upar, uperp)
    𝐱 = 𝐡.-𝐡_hh
    𝐇_hh = dualHH(R, A, B, ϕCapitalpar, ϕCapitalperp, Upar2, Uperp2)
    𝐗 = 𝐡.-𝐇_hh

    @show maximum(norm.(𝐱[jⁱⁿ]))
    @show maximum(norm.(𝐡[jⁱⁿ]))
    @show maximum(norm.(𝐗[jⁱⁿ]))
    # @show maximum(norm.(𝐇[jⁱⁿ]))

    gl = GridLayout(fig[1,col*2])
    
    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:I 
        poly!(axes[end], cellPolygons[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
    end
    # arrowColours = [(:blue, norm(𝐱[j])/maximum(norm.(𝐱[jⁱⁿ]))) for j=1:J]
    arrowColours = [(:blue, norm(𝐱[j])/0.0464) for j=1:J]
    arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ)[jⁱⁿ], Vec{2,Float64}.(𝐱[jⁱⁿ]), color=arrowColours[jⁱⁿ], linewidth=2, lengthscale=20.0)
    # arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ)[jⁱⁿ], Vec{2,Float64}.(𝐱[jⁱⁿ]), color=:red, linewidth=2, lengthscale=10.0)
    Label(gl[1,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for i=1:I 
        poly!(axes[end], cellPolygons[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
    end
    # arrowColours = [(:green, norm(𝐗[j])/maximum(norm.(𝐗[jⁱⁿ]))) for j=1:J]
    arrowColours = [(:green, norm(𝐗[j])/0.0464) for j=1:J]
    arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ)[jⁱⁿ], Vec{2,Float64}.(𝐗[jⁱⁿ]), color=arrowColours[jⁱⁿ], linewidth=2, lengthscale=20.0)
    # arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ)[jⁱⁿ], Vec{2,Float64}.(𝐗[jⁱⁿ]), color=:red, linewidth=2, lengthscale=10.0)
    Label(gl[2,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 


    rowsize!(gl, 1, Relative(0.5))
    rowsize!(gl, 2, Relative(0.5))
    # rowsize!(gl, 3, Relative(0.48))
    # rowsize!(gl, 4, Relative(0.02))

end

gl = GridLayout(fig[1,1])
Label(gl[1,1], "Primal", fontsize=24) 
Label(gl[2,1], "Dual", fontsize=24) 
rowsize!(gl, 1, Relative(0.5))
rowsize!(gl, 2, Relative(0.5))

gl = GridLayout(fig[2,1])
push!(axes, Axis(fig[:,3]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
push!(axes, Axis(fig[:,5]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)

hidedecorations!.(axes)
hidespines!.(axes)

rowsize!(fig.layout, 1, Relative(0.99))
rowsize!(fig.layout, 2, Relative(0.01))
colsize!(fig.layout, 1, Relative(0.05))
colsize!(fig.layout, 2, Relative(0.31))
colsize!(fig.layout, 3, Relative(0.01))
colsize!(fig.layout, 4, Relative(0.31))
colsize!(fig.layout, 5, Relative(0.01))
colsize!(fig.layout, 6, Relative(0.31))

resize_to_layout!(fig)

display(fig)
save(plotsdir(inputDir, "Figure9HelmholtzHodgeHoles.png"), fig)
save(plotsdir(inputDir, "Figure9HelmholtzHodgeHoles.pdf"), fig)
