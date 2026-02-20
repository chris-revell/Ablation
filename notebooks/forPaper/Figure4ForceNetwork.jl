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
using Statistics

# function num_to_string(x, fmt="%.1g")
function numToString(x, fmt="%.1e")
    Printf.format(Printf.Format(fmt), x)
end

getRandomColor(seed) = RGB(rand(Xoshiro(seed),3)...)

pExt = [0.5, 0.0]

fig = Figure(size=(1500, 1000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

# if !(isfile(datadir("referenceSystems", "Figure4", "row1.jld2")) && isfile(datadir("referenceSystems", "Figure4", "row2.jld2")))
#     integ = vertexModel(nRows = 9,
#                             # pressureExternal=pExt[row],
#                             nCycles=2.0,
#                             outputToggle=0,
#                             frameDataToggle=0,
#                             frameImageToggle=0,
#                             videoToggle=0,
#                             printToggle=1,
#                             energyModel="quadratic",
#                             divisionToggle=1
#                         )

#         params, matrices = integ.p
#         Rtmp = reinterpret(SVector{2,Float64}, integ.u)
#         Atmp = matrices.A
#         Btmp = matrices.B
#         Ftmp = matrices.F
# end


for row = 1:2

    gl = GridLayout(fig[row,:])

    if isfile(datadir("referenceSystems", "Figure4", "row$(row).jld2"))
        importedData = load(datadir("referenceSystems", "Figure4", "row$(row).jld2"))
        @unpack R, A, B, F = importedData
    else
        integ = vertexModel(nRows = 9,
                            # pressureExternal=pExt[row],
                            nCycles=2.0,
                            outputToggle=0,
                            frameDataToggle=0,
                            frameImageToggle=0,
                            videoToggle=0,
                            printToggle=1,
                            energyModel="quadratic",
                            divisionToggle=1
                        )

        params, matrices = integ.p
        Rtmp = reinterpret(SVector{2,Float64}, integ.u)
        Atmp = matrices.A
        Btmp = matrices.B
        Ftmp = matrices.F
        integ2 = vertexModel(abstol = 1e-9,
                            reltol = 1e-9,
                            initialSystem="argument",
                            divisionToggle=0,
                            R_in=Rtmp,
                            A_in=Atmp,
                            B_in=Btmp,
                            pressureExternal=pExt[row],
                            nCycles=1.0,
                            outputToggle=0,
                            frameDataToggle=0,
                            frameImageToggle=0,
                            videoToggle=0,
                            printToggle=1,
                            energyModel="quadratic",
                        )
        params2, matrices2 = integ2.p

        R = reinterpret(SVector{2,Float64}, integ2.u)
        A = matrices2.A
        B = matrices2.B
        F = matrices2.F

        !isdir(datadir("referenceSystems", "Figure4")) ? mkpath(datadir("referenceSystems", "Figure4")) : nothing
        jldsave(datadir("referenceSystems", "Figure4", "row$(row).jld2"); 
            R,
            A, 
            B, 
            F, 
        )
    end

    I = size(B,1)
    J = size(B,2)
    K = size(A,2)
    kᵖ = findPeripheralVertices(A, B)
    kⁱ = 1 .- kᵖ
    Kⁱ = sum(kⁱ)
    iᵖ = findPeripheralCells(B)
    linkTriangles = findCellLinkTriangles(R, A, B)[kⁱ.==1]
    cellPolygons = findCellPolygons(R, A, B)
    cellPositions = findCellCentresOfMass(R, A, B)
    𝐬ᵢₖ = findEdgeMidpointLinks(R, A, B)
    𝐜ⱼ = findEdgeMidpoints(R, A)
    𝐡 = hNetwork(R, A, B, F)
    cellVertexOrders  = fill(CircularVector(Int64[]), I)
    cellEdgeOrders    = fill(CircularVector(Int64[]), I)
    for i = 1:I
        cellVertexOrders[i], cellEdgeOrders[i] = orderAroundCell(A, B, i)
    end
    ϵ = SMatrix{2, 2, Float64}([
            0.0 1.0
            -1.0 0.0
        ])

    centralCell = 5
    cellNeighbourMatrix = B*transpose(B)
    centralNeighbours = findall(x->x!=0, cellNeighbourMatrix[5,:])
    neighbours = []
    for n in centralNeighbours
        localNeighbours = findall(x->x!=0, cellNeighbourMatrix[n,:])
        append!(neighbours, localNeighbours)
    end
    unique!(neighbours)
    neighbours = centralNeighbours

    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:I
        if i∈neighbours
            poly!(axes[end],
                cellPolygons[i],
                color=(getRandomColor(i), 0.5),
                strokewidth=1,
                strokecolor=(:black, 1.0)
            )
        else
            poly!(axes[end],
                cellPolygons[i],
                color=(getRandomColor(i), 0.0),
                strokewidth=1,
                strokecolor=(:black, 1.0)
            )
        end
    end
    # scatter!(axes[end], Point{2,Float64}.(cellPositions), color=:black)
    Label(gl[1,1, Bottom()], popfirst!(subfigureLabels))

    push!(axes, Axis(gl[1,2], aspect=DataAspect()))
    for i=1:I
        if i∈neighbours
            poly!(axes[end],
                cellPolygons[i],
                color=(getRandomColor(i), 0.25),
                strokewidth=1,
                strokecolor=(:black, 1.0)
            )
            lines!(
                axes[end],
                Point{2,Float64}.(𝐜ⱼ[cellEdgeOrders[i][0:end]]),
                color=(getRandomColor(i), 1.0),
                linewidth=4,
            )
            scatter!(axes[end], Point{2,Float64}.(𝐜ⱼ[cellEdgeOrders[i]]), color=:black)
            # annotation!(axes[end], Point{2,Float64}.(𝐜ⱼ[cellEdgeOrders[i]]), text = string.(cellEdgeOrders[i]), fontsize=12)
        end
    end
    if row == 1
        annotation!(axes[end], Point{2,Float64}(𝐜ⱼ[35].+[15.0,15.0]), Point{2,Float64}(𝐜ⱼ[35]), text = L"\mathbf{c}_j", fontsize=36)
    elseif row == 2
        annotation!(axes[end], Point{2,Float64}(𝐜ⱼ[586].+[15.0,15.0]), Point{2,Float64}(𝐜ⱼ[586]), text = L"\mathbf{h}_j", fontsize=36)
    end 
    Label(gl[1,2, Bottom()], popfirst!(subfigureLabels))


    push!(axes, Axis(gl[1,3], aspect=DataAspect()))
    Label(gl[1,3, Bottom()], popfirst!(subfigureLabels))
    for i in neighbours
        # lines!(axes[end], Point{2,Float64}.(𝐡[cellEdgeOrders[i][0:end]]), color=:black)
        rotatedForces = [ϵ*F[k, i] for k in cellVertexOrders[i][0:end-1]]
        arrows2d!(axes[end], Point{2,Float64}.(𝐡[cellEdgeOrders[i]]), Vec{2,Float64}.(rotatedForces), color=getRandomColor(i), align=:tip)
    end
    if row == 1
        annotation!(axes[end], Point{2,Float64}(𝐡[35].+[15.0,15.0]), Point{2,Float64}(𝐡[35]), text = L"\mathbf{h}_j", fontsize=36)
    elseif row == 2
        annotation!(axes[end], Point{2,Float64}(𝐡[586].+[15.0,15.0]), Point{2,Float64}(𝐡[586]), text = L"\mathbf{h}_j", fontsize=36)
    end 
end


hidedecorations!.(axes)
hidespines!.(axes)
save(plotsdir("Figure4ForceNetwork.png"), fig)
save(plotsdir("Figure4ForceNetwork.pdf"), fig)

display(fig)
