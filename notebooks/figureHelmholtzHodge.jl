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

inputSystems = ["NoHole", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

suppressOrNot = "nosuppress"
spokesOrNot = "nospokes"

fig = Figure(size=(1500, 1500))
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"($l)" for l in individualLetters]

for (col,inputSystem) in enumerate(inputSystems)
    # Import system data
    if inputSystem == "OldSystem"
        # conditionsDict    = load(datadir("referenceSystems", "oldPaper", "dataFinal.jld2"))
        # @unpack  = conditionsDict["params"]
        matricesDict = load(datadir("referenceSystems", "oldPaper", "matricesFinal.jld2"))
        @unpack A,B,C,R,F,cellAreas,cellPressures,cellTensions,cellPerimeters = matricesDict["matrices"]
        cellEffectivePressures = cellPressures .- cellTensions.*cellPerimeters./(2.0.*cellAreas)
        dropzeros!(A)
        dropzeros!(B)
        dropzeros!(C)
    else 
        fileName = datadir("referenceSystems", "$(inputSystem)_testSystem5.jld2")
        importedData = load(fileName)
        R = importedData["R"]
        A = importedData["A"]
        B = importedData["B"]
        F = importedData["F"]
        cellTensions = importedData["cellTensions"]
        cellPressures = importedData["cellPressures"]
        cellPerimeters = importedData["cellPerimeters"]
        cellAreas = importedData["cellAreas"]
        cellEffectivePressures = cellPressures .+ cellTensions.*cellPerimeters./(2.0.*cellAreas)
    end
    
    I = size(B,1)
    J = size(B,2)
    K = size(A,2)
    cellAreas = findCellAreas(R, A, B)
    linkTriangles = findCellLinkTriangles(R, A, B)
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)
    # 𝐡 = [SVector{2,Float64}(1.0,0.0) for _=1:J]

    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᶜhMax = max(maximum(abs.(curlᶜh)), 0.0001)
    curlᵛh = (spokesOrNot=="spokes" ? curlᵛspokes(R, A, B, 𝐡) : curlᵛ(R, A, B, 𝐡))
    curlᵛhMax = max(maximum(abs.(curlᵛh)), 0.0001)
    divᶜh = (suppressOrNot=="suppress" ? divᶜsuppress(R, A, B, 𝐡) : divᶜ(R, A, B, 𝐡))
    divᶜhMax = max(maximum(abs.(divᶜh)), 0.0001)
    divᵛh = divᵛ(R, A, B, 𝐡)
    divᵛhMax = max(maximum(abs.(divᵛh)), 0.0001)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᶜhMax = max(maximum(abs.(cocurlᶜh)), 0.0001)
    cocurlᵛh = (spokesOrNot=="spokes" ? cocurlᵛspokes(R, A, B, 𝐡) : cocurlᵛ(R, A, B, 𝐡))
    cocurlᵛhMax = max(maximum(abs.(cocurlᵛh)), 0.0001)
    codivᶜh = (suppressOrNot=="suppress" ? codivᶜsuppress(R, A, B, 𝐡) : codivᶜ(R, A, B, 𝐡))
    codivᶜhMax = max(maximum(abs.(codivᶜh)), 0.0001)
    codivᵛh = codivᵛ(R, A, B, 𝐡)
    codivᵛhMax = max(maximum(abs.(codivᵛh)), 0.0001)
    
    H = Diagonal(cellAreas)
    E = Diagonal(linkTriangleAreas)
    Lv = geometricLv(R, A, B)
    Lf = geometricLf(R, A, B)
    Lc = geometricLc(R, A, B)
    Lt = geometricLt(R, A, B)
    # ϕpar Lv -divᵛ
    ϕpar, ϕparspectrum = penrosePseudoInversion(Lv, -1.0.*divᵛh, E)
    # ϕperp Lv -codivᵛ
    ϕperp, ϕperpspectrum = penrosePseudoInversion(Lv, -1.0.*codivᵛh, E)
    # upar Lf cocurlᶜ
    upar, uparspectrum = penrosePseudoInversion(Lf, cocurlᶜh, H)
    # uperp Lf curlᶜ
    uperp, uperpspectrum = penrosePseudoInversion(Lf, curlᶜh, H)
    # ϕCapitalpar Lc -divᶜ
    ϕCapitalpar, ϕCapitalparspectrum = penrosePseudoInversion(Lc, -1.0.*divᶜh, H)
    # ϕCapitalperp Lc -codivᶜ
    ϕCapitalperp, ϕCapitalperpspectrum = penrosePseudoInversion(Lc, -1.0.*codivᶜh, H)
    # Upar Lt cocurlᵛ
    Upar, Uparspectrum = penrosePseudoInversion(Lt, cocurlᵛh, E)
    # Uperp Lt curlᵛ
    Uperp, Uperpspectrum = penrosePseudoInversion(Lt, curlᵛh, E)

    # 𝐯 = grad ϕ + rot u + x 
    # 𝐕 = grad ϕCapital + rot U + x 
    # grad ϕ = gradᵛ ϕpar + cogradᵛ ϕperp 
    # grad ϕCapital = gradᶜ ϕCapitalpar + cogradᶜ ϕCapitalperp 
    # rot u = rotᶜ uperp + corotᶜ upar 
    # rot U = rotᵛ Uperp + corotᵛ Upar

    𝐡_hh = gradᵛ(R, A, ϕpar) + cogradᵛ(R, A, B, ϕperp) + rotᶜ(R, A, B, uperp) + corotᶜ(R, A, B, upar) 
    𝐇_hh = gradᶜ(R, A, B, ϕCapitalpar) + cogradᶜ(R, A, B, ϕCapitalperp) + rotᵛspokes(R, A, B, Uperp) + corotᵛspokes(R, A, B, Upar)

    push!(axes, Axis(fig[1,col], aspect=DataAspect()))
    for i=1:size(B,1)
        poly!(axes[end],cellPolygons[i],color=(:white, 1.0),strokewidth=1,strokecolor=(:black,1.0))
    end
    hidedecorations!(axes[end])
    hidespines!(axes[end])
    
    push!(axes, Axis(fig[2,col], aspect=DataAspect()))
    scatter!(axes[end], Point{2,Float64}.(𝐡_hh), color=(:red, 0.55))
    scatter!(axes[end], Point{2,Float64}.(𝐇_hh), color=(:green, 0.55))
    Label(fig[2,col, Bottom()], "h")

    push!(axes, Axis(fig[3,col], aspect=DataAspect()))
    scatter!(axes[end], Point{2,Float64}.(𝐡_hh.-𝐡), color=(:red, 0.55))
    scatter!(axes[end], Point{2,Float64}.(𝐇_hh.-𝐡), color=(:green, 0.55))
    Label(fig[3,col, Bottom()], "x")
end

display(fig)
save(plotsdir("figureHelmholtz.png"), fig)