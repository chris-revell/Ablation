# Old: Lf ψᶜ = -divᶜ 𝐡 
# Conversion: -divᶜ => cocurlᶜ
# New: Lf ψᶜ = cocurlᶜ 𝐡 

#  Old: Lt ψ̆ᵛ = -divᵛ h̆
#  Conversion: -divᵛ => cocurlᵛ
#  New: Lt ψ̆ᵛ = cocurlᵛ h̆

#  Old: Lt Ψ̆ᵛ = CURLᵛ h̆
#  Conversion: CURLᵛ => -curlᵛ
#  New: Lt Ψ̆ᵛ = -curlᵛ h̆

# RHS:
# -divᶜ => cocurlᶜ
# -divᵛ => cocurlᵛ
# CURLᵛ => -curlᵛ


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

# integ1 = vertexModel(nRows=11, nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0)
# R = reinterpret(SVector{2,Float64}, integ1.u) 
# params, matrices = integ1.p
# # integ = vertexModel(initialSystem="argument", R_in=R, A_in=matrices.A, B_in=matrices.B, nCycles=0.1, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0, divisionToggle=0)
# # R = reinterpret(SVector{2,Float64}, integ.u) 
# # params, matrices = integ.p
# @unpack A, B, F, cellPressures, cellTensions, cellPerimeters, cellAreas = matrices

# # Import system data
# conditionsDict    = load(datadir("oldPaper", "dataFinal.jld2"))
# @unpack nVerts,nCells,nEdges,pressureExternal,γ,λ,viscousTimeScale,realTimetMax,tMax,dt,outputInterval,outputTotal,realCycleTime,t1Threshold = conditionsDict["params"]
# matricesDict = load(datadir("oldPaper", "matricesFinal.jld2"))
# @unpack A,B,C,R,F,cellAreas,cellPressures,cellTensions,cellPerimeters = matricesDict["matrices"]
# cellEffectivePressures = cellPressures .- cellTensions.*cellPerimeters./(2.0.*cellAreas)
# dropzeros!(A)
# dropzeros!(B)
# dropzeros!(C)

fileName = datadir("25-05-14-12-18-35_testSystem.jld2")
importedData = load(fileName)
R = importedData["ablatedGrownR"]
A = importedData["ablatedGrownA"]
B = importedData["ablatedGrownB"]
F = importedData["ablatedGrownF"]
cellTensions = importedData["ablatedGrownCellTensions"]
cellPressures = importedData["ablatedGrownCellPressures"]
cellPerimeters = importedData["ablatedGrownCellPerimeters"]
cellAreas = importedData["ablatedGrownCellAreas"]
cellEffectivePressures = cellPressures .+ cellTensions.*cellPerimeters./(2.0.*cellAreas)

function eigenmodeSolve(L, RHS, metricMatrix)
    n = length(RHS)
    eigenvectors = (eigen(Matrix(L))).vectors
    eigenvalues = (eigen(Matrix(L))).values
    onesVec = ones(n)
    ḡ = ((onesVec'*metricMatrix*RHS)/(onesVec'*metricMatrix*onesVec)) .*onesVec
    ğ = RHS.-ḡ
    ψ̆ = zeros(n)
    ψ̆Spectrum = Float64[]
    for k=2:n
        numerator = eigenvectors[:,k]'*metricMatrix*ğ
        denominator = eigenvalues[k]*(eigenvectors[:,k]'*metricMatrix*eigenvectors[:,k])

        #!!!!!!!!! - sign ? 
        ψ̆ .-= (numerator/denominator).*eigenvectors[:,k]
        #!!!!!!!!! - sign ? 

        push!(ψ̆Spectrum,(numerator/denominator))
    end
    return ψ̆, ψ̆Spectrum
end

# Figure 6 in old paper effectively solved
# ψc = eigenmodeSolve(Lf, -1.0.*calculateCellDivs(), H)
# ψ̆ᵥ = eigenmodeSolve(Lt, -1.0.*calculateVertexDivs(), E)
# Ψ̆ᵥ = eigenmodeSolve(Lt, calculateVertexCurls(), E)
# old vs new operators demonstrated that:
# calculateCellDivs() => -cocurlᶜh = divᶜ
# calculateVertexDivs() => -cocurlᵛh = divᵛ
# calculateVertexCurls() => -curlᵛ = CURLᵛ
# Therefore everything in the old paper was correct ASSUMING THAT THE SIGN IN THE PENROSE INVERSION IS CORRECT



nCells = size(B,1)
nEDges = size(B,2)
nVerts = size(A,2)
cellAreas = findCellAreas(R, A, B)
linkTriangles = findCellLinkTriangles(R, A, B)
linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
cellPolygons = findCellPolygons(R, A, B)
𝐡 = hNetwork(R, A, B, F)


Lf = geometricLf(R, A, B)
# Lf ψc = -divᶜ 𝐡
# -divᶜ => cocurlᶜ
# Lf ψᶜ = cocurlᶜ 𝐡
cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
H = Diagonal(cellAreas)
ψc, ψcspectrum = eigenmodeSolve(Lf, cocurlᶜh, H)
cocurlᶜhLims = (-maximum(abs.(cocurlᶜh)), maximum(abs.(cocurlᶜh)))
cellPEffLims = 2.0.*(-maximum(abs.(cellEffectivePressures)), maximum(abs.(cellEffectivePressures)))
ψcLims = (-maximum(abs.(ψc)),maximum(abs.(ψc)))

curlᶜh = curlᶜ(R, A, B, 𝐡)
curlᶜhLims = (-maximum(abs.(curlᶜh)), maximum(abs.(curlᶜh)))
# curlᶜhLims = (-0.001, 0.001)

# 𝐡̆ = 𝐡
# Lt ψ̆ᵥ = -divᵛ 𝐡̆
# -divᵛ => cocurlᵛ
# Lt ψ̆ᵛ = cocurlᵛ h̆
Lₜ = geometricLt(R, A, B)
cocurlᵛh = cocurlᵛ(R, A, B, 𝐡)
E = Diagonal(linkTriangleAreas)
ψ̆ᵥ, ψ̆ᵥspectrum = eigenmodeSolve(Lₜ, cocurlᵛh, E)
cocurlᵛhLims = (-maximum(abs.(cocurlᵛh)),maximum(abs.(cocurlᵛh)))
ψ̆ᵥLims = (-maximum(abs.(ψ̆ᵥ)),maximum(abs.(ψ̆ᵥ)))

# Lₜ Ψ̆ᵥ = CURLᵛ 𝐡
# CURLᵛ => -curlᵛ
# Lt Ψ̆ᵛ = -curlᵛ h̆
Lₜ = geometricLt(R, A, B)
curlᵛh = curlᵛ(R, A, B, 𝐡)
E = Diagonal(linkTriangleAreas)
Ψ̆ᵥ, Ψ̆ᵥspectrum = eigenmodeSolve(Lₜ, -1.0.*curlᵛh, E)
curlᵛLims = (-maximum(abs.(curlᵛh)),maximum(abs.(curlᵛh)))
Ψ̆ᵥLims = (-maximum(abs.(Ψ̆ᵥ)),maximum(abs.(Ψ̆ᵥ)))


#%%

fig = Figure(size=(1200,900))

#ax11 
ax11 = Axis(fig[1,1], aspect=DataAspect())
for i=1:nCells
    poly!(ax11,cellPolygons[i],color=cocurlᶜh[i],colorrange=cocurlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
end
hidedecorations!(ax11)
hidespines!(ax11)
Colorbar(fig[1,2], colorrange=cocurlᶜhLims, colormap=:bwr)
Label(fig[1,1,Bottom()],L"\{cocurl^c h\}_i",fontsize = 24)
#ax12 
ax12 = Axis(fig[1,3], aspect=DataAspect())
for i=1:nCells
    poly!(ax12,cellPolygons[i],color=2.0.*cellEffectivePressures[i],colorrange=cellPEffLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
end
hidedecorations!(ax12)
hidespines!(ax12)
Colorbar(fig[1,4], colorrange=cellPEffLims, colormap=:bwr)
Label(fig[1,3,Bottom()],L"2 \times P_{eff}",fontsize = 24)
#ax13
ax13 = Axis(fig[1,5-0],aspect=DataAspect())
hidedecorations!(ax13)
hidespines!(ax13)
for i=1:nCells
    poly!(ax13,cellPolygons[i],color=ψc[i],colorrange=ψcLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[1,6-0], colorrange=ψcLims, colormap=:bwr)
Label(fig[1,5-0,Bottom()],L"\psi_c",fontsize = 24)
#ax14
ax14 = Axis(fig[1,7-0], xlabel=L"i", ylabel=L"log_{10}(Amplitude)", alignmode = Outside())
xlims!(ax14,0,nCells)
# ylims!(ax14,0,1.1*maximum(log.(abs.(ψcspectrum))))
# barplot!(ax14,collect(2:nCells),log.(abs.(ψcspectrum)),width=1.0,color=:blue,strokecolor=:blue)
lines!(ax14,collect(2:nCells),log.(abs.(ψcspectrum)))
Label(fig[1,7-0,Bottom()],L"\psi_c Spectrum",fontsize = 24)



#ax21 
ax21 = Axis(fig[2,1], aspect=DataAspect())
for i=1:nCells
    poly!(ax21,cellPolygons[i],color=curlᶜh[i],colorrange=curlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
end
hidedecorations!(ax21)
hidespines!(ax21)
Colorbar(fig[2,2], colorrange=curlᶜhLims, colormap=:bwr)
Label(fig[2,1,Bottom()],L"\{curl^c h\}_i",fontsize = 24)



#ax31
ax31 = Axis(fig[3,1],aspect=DataAspect())
hidedecorations!(ax31)
hidespines!(ax31)
for k=1:nVerts
    poly!(ax31,linkTriangles[k],color=cocurlᵛh[k],colorrange=cocurlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax31,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[3,2],limits=cocurlᵛhLims,colormap=:bwr)
Label(fig[3,1,Bottom()], L"\{cocurl^v \breve{h}\}_k", fontsize = 24)
#ax32
#ax33
ax33 = Axis(fig[3,5-0],aspect=DataAspect())
hidedecorations!(ax33)
hidespines!(ax33)
for k=1:nVerts
    poly!(ax33,linkTriangles[k],color=ψ̆ᵥ[k],colorrange=ψ̆ᵥLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax33,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[3,6-0], colorrange=ψ̆ᵥLims, colormap=:bwr)
Label(fig[3,5-0,Bottom()],L"\breve{\psi}_v",fontsize = 24)
#ax34
ax34 = Axis(fig[3,7-0], xlabel=L"k", ylabel=L"log_{10}(Amplitude)", alignmode = Outside())
xlims!(ax34,0,nVerts)
# ylims!(ax34,0,1.1*maximum(log.(abs.(ψ̆ᵥspectrum))))
# barplot!(ax34,collect(2:nVerts),log.(abs.(ψ̆ᵥspectrum)),width=1.0,color=:orange,strokecolor=:orange)
lines!(ax34,collect(2:nVerts),log.(abs.(ψ̆ᵥspectrum)))
Label(fig[3,7-0,Bottom()],L"\breve{\psi}_v Spectrum",fontsize = 24)

#ax41
ax41 = Axis(fig[4,1],aspect=DataAspect())
hidedecorations!(ax41)
hidespines!(ax41)
for k=1:nVerts
    poly!(ax41,linkTriangles[k],color=-curlᵛh[k],colorrange=curlᵛLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax41,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[4,2],limits=curlᵛLims,colormap=:bwr)
Label(fig[4,1,Bottom()], L"-\{curl^v \breve{h}\}_k", fontsize = 24)

#ax42


#ax43
ax43 = Axis(fig[4,5-0],aspect=DataAspect())
hidedecorations!(ax43)
hidespines!(ax43)
for k=1:nVerts
    poly!(ax43,linkTriangles[k],color=Ψ̆ᵥ[k],colorrange=Ψ̆ᵥLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax43,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[4,6-0], colorrange=Ψ̆ᵥLims, colormap=:bwr)
Label(fig[4,5-0,Bottom()],L"\breve{\Psi}_v",fontsize = 24)

#ax44
ax44 = Axis(fig[4,7-0], xlabel=L"k", ylabel=L"log_{10}(Amplitude)", alignmode = Outside())
xlims!(ax44,0,nVerts)
# ylims!(ax44,0,1.1*maximum(log.(abs.(Ψ̆ᵥspectrum))))
# barplot!(ax44,collect(2:nVerts),log.(abs.(Ψ̆ᵥspectrum)),width=1.0,color=:orange,strokecolor=:orange)
lines!(ax44,collect(2:nVerts),log.(abs.(Ψ̆ᵥspectrum)))
Label(fig[4,7-0,Bottom()],L"\breve{\Psi}_v Spectrum",fontsize = 24)

display(fig)
save(datadir("$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))_potentials.png"), fig)

#%%

# ℒₓ = A*transpose(A) + transpose(B)*B

# eigenvectors = (eigen(Matrix(ℒₓ))).vectors
# eigenvalues = (eigen(Matrix(ℒₓ))).values

# @show eigenvalues[1]


# edgeQuadrilaterals = findEdgeQuadrilaterals(R, A, B)

# fig = Figure(size=(500,500))
# ax = Axis(fig[1,1], aspect=DataAspect())

# crange = (minimum(eigenvectors[:,1]), maximum(eigenvectors[:,1]))
# for i=1:size(A,1)
#     poly!(ax, edgeQuadrilaterals[i], color=(eigenvectors[i,1]), colorrange=crange, colormap=:inferno)
# end
# hidedecorations!(ax)
# hidespines!(ax)
# Colorbar(fig[1,2], colorrange=crange, colormap=:inferno)
# display(fig)





# function psicPotential(R, A, B, 𝐛)
#     Lf = geometricLf(R, A, B)
#     cellDivs = cocurlᶜ(R, A, B, 𝐛)
#     H = Diagonal(findCellAreas(R, A, B))
#     ψ̆, spectrum = eigenModeSolve(Lf, H, cellDivs)
#     return ψ̆, spectrum
# end

# function psivPotential(R, A, B, 𝐛)
#     Lₜ = geometricLt(R, A, B)
#     vertexDivs = cocurlᵛ(R, A, B, 𝐛)
#     E = Diagonal(findCellLinkTriangleAreas(R, A, B))
#     ψ̆, spectrum = eigenModeSolve(Lₜ, E, vertexDivs)
#     return ψ̆, spectrum
# end

# function capitalPsivPotential(R, A, B, 𝐛)
#     Lₜ = geometricLt(R, A, B)
#     vertexCurls = curlᵛ(R, A, B, 𝐛)
#     E = Diagonal(findCellLinkTriangleAreas(R, A, B))
#     ψ̆, spectrum = eigenModeSolve(Lₜ, E, vertexCurls)
#     return ψ̆, spectrum
# end

# function potential(ℒ, deriv, metric)
#     Lₜ = geometricLt(R, A, B)
#     vertexCurls = curlᵛ(R, A, B, 𝐛)
#     E = Diagonal(findCellLinkTriangleAreas(R, A, B))
#     ψ̆, spectrum = eigenModeSolve(Lₜ, E, vertexCurls)
#     return ψ̆, spectrum
# end

# # {divᶜb}ᵢ
# # Calculate div on each cell
# function calculateCellDivs(nCells, R, B, C, F, cellCentresOfMass, edgeMidpoints, edgeTangents, cellAreas, ϵᵢ)
#     cellDivs = Float64[]
#     for c = 1:nCells
#         cellVertices = findall(x -> x != 0, C[c, :])
#         vertexAngles = zeros(size(cellVertices))
#         for (k, v) in enumerate(cellVertices)
#             vertexAngles[k] = atan((R[v] .- cellCentresOfMass[c])...)
#         end
#         m = minimum(vertexAngles)
#         vertexAngles .-= m
#         cellVertices .= cellVertices[sortperm(vertexAngles)]
#         cellEdges = findall(x -> x != 0, B[c, :])
#         edgeAngles = zeros(size(cellEdges))
#         for (k, e) in enumerate(cellEdges)
#             edgeAngles[k] = atan((edgeMidpoints[e] .- cellCentresOfMass[c])...)
#         end
#         edgeAngles .+= (2π - m)
#         edgeAngles .= edgeAngles .% (2π)
#         cellEdges .= cellEdges[sortperm(edgeAngles)]
#         h = @SVector [0.0, 0.0]
#         divSum = 0
#         for (i, e) in enumerate(cellEdges)
#             h = h + ϵᵢ * F[cellVertices[i], c]
#             divSum -= B[c, e] * (h ⋅ (ϵᵢ * edgeTangents[e])) / cellAreas[c]
#         end
#         # divSum *= (-0.5)
#         push!(cellDivs, divSum)
#     end
#     return cellDivs
# end

# # {divᵛb}ₖ
# # Calculate div at each vertex
# function calculateVertexDivs(nVerts, R, C, cellCentresOfMass, F, ϵᵢ, q, linkTriangleAreas)
#     vertexDivs = Float64[]
#     for k = 1:nVerts
#         divSum = 0
#         vertexCells = findall(x -> x != 0, C[:, k])
#         cellAngles = zeros(length(vertexCells))
#         for i = 1:length(cellAngles)
#             cellAngles[i] = atan((cellCentresOfMass[vertexCells[i]] .- R[k])...)
#         end
#         vertexCells .= vertexCells[sortperm(cellAngles, rev=true)]
#         for i in vertexCells
#             divSum += ((ϵᵢ * q[i, k]) ⋅ (ϵᵢ * F[k, i])) / linkTriangleAreas[k]
#         end
#         push!(vertexDivs, divSum)
#     end
#     return vertexDivs
# end

# # {CURLᵛb}ₖ
# # Calculate curl at each vertex
# function calculateVertexCurls(nVerts, R, C, cellCentresOfMass, F, ϵᵢ, q, linkTriangleAreas)
#     vertexCurls = Float64[]
#     # Working around a given vertex, an h force space point from a cell is mapped to the next edge anticlockwise from the cell
#     for k = 1:nVerts
#         curlSum = 0
#         vertexCells = findall(x -> x != 0, C[:, k])
#         cellAngles = zeros(length(vertexCells))
#         for i = 1:length(cellAngles)
#             cellAngles[i] = atan((cellCentresOfMass[vertexCells[i]] .- R[k])...)
#         end
#         vertexCells .= vertexCells[sortperm(cellAngles, rev=true)]
#         for i in vertexCells
#             curlSum += (q[i, k] ⋅ (ϵᵢ * F[k, i])) / linkTriangleAreas[k]
#         end
#         push!(vertexCurls, curlSum)
#     end
#     return vertexCurls
# end