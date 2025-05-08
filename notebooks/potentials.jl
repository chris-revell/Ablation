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

integ1 = vertexModel(nRows=11, nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0)
R = reinterpret(SVector{2,Float64}, integ1.u) 
params, matrices = integ1.p
integ = vertexModel(initialSystem="argument", R_in=R, A_in=matrices.A, B_in=matrices.B, nCycles=0.1, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0, divisionToggle=0)
R = reinterpret(SVector{2,Float64}, integ.u) 
params, matrices = integ.p
@unpack A, B, F = matrices

nCells = size(B,1)
nEDges = size(B,2)
nVerts = size(A,2)

#%%

# ℒₓ = ablatedGrownA*transpose(ablatedGrownA) + transpose(ablatedGrownB)*ablatedGrownB
# eigenvectorsℒₓ = (eigen(Matrix(ℒₓ))).vectors
# eigenvalues = (eigen(Matrix(ℒₓ))).values
# cellPolygons = findCellPolygons(matrices...)
# edgeQuadrilaterals = findEdgeQuadrilaterals(matrices...)

#%%


# psi_c potential axis
Lf = geometricLf(R, A, B)


𝒜 = spdiagm(1.0./findCellAreas(R, A, B))
𝐭 = spdiagm(findEdgeTangents(R, A))
# 𝐭2 = spdiagm(transpose.(findEdgeTangents(R, A)))
ϵᵢ = SMatrix{2, 2, Float64}([
            0.0 1.0
            -1.0 0.0
        ])
ϵ = spdiagm(fill(ϵᵢ, nCells))

h = hNetwork(R, A, B, F)

divᶜᵢ = 𝒜*ϵ*B*𝐭

divVals = [sum(divᶜᵢ[i,:].⋅h) for i=1:nCells]

#%%

fig = Figure(size=(500,1000))
ax = Axis(fig[1,1], aspect=DataAspect())
cellPolygons = findCellPolygons(R, A, B)
crange = (minimum(divVals), maximum(divVals))
for i=1:nCells 
    poly!(ax, cellPolygons[i], color=divVals[i], colorrange=crange, colormap=:bwr, strokecolor=(:black, 0.5), strokewidth=2)
end
hidedecorations!(ax)
hidespines!(ax)
Colorbar(fig[1,2], colorrange=crange, colormap=:bwr)
display(fig)


cellAreas = findCellAreas(R, A, B)

onesVec = ones(nCells)
H = spdiagm(cellAreas)
eigenvectors = (eigen(Matrix(Lf))).vectors
eigenvalues = (eigen(Matrix(Lf))).values
ḡ = ((onesVec'*H*divVals)/(onesVec'*H*ones(nCells))).*onesVec
ğ = divVals.-ḡ
ψ̆ = zeros(nCells)
eigenmodeAmplitudes = Float64[]
for k=2:nCells
    numerator = eigenvectors[:,k]'*H*ğ
    denominator = eigenvalues[k]*(eigenvectors[:,k]'*H*eigenvectors[:,k])
    ψ̆ .-= (numerator/denominator).*eigenvectors[:,k]
    push!(eigenmodeAmplitudes,(numerator/denominator))
end
ψ̆Lims = (-maximum(abs.(ψ̆)),maximum(abs.(ψ̆)))
ax1 = Axis(fig[2,1],aspect=DataAspect())
hidedecorations!(ax1)
hidespines!(ax1)
for i=1:nCells
    poly!(ax1, cellPolygons[i], color=ψ̆[i], colormap=:bwr, colorrange=ψ̆Lims, strokecolor=(:black,1.0), strokewidth=1)
end
Colorbar(fig[2,2], colorrange=ψ̆Lims, colormap=:bwr)
display(fig)



#%%

# psi_c spectrum axis
ax2 = Axis(fig[1,2], xlabel="Eigenmode number, i", ylabel="Amplitude", alignmode = Outside())
xlims!(ax2,0,nCells)
ylims!(ax2,0,1.1*maximum(abs.(eigenmodeAmplitudes)))
barplot!(ax2,collect(2:nCells),abs.(eigenmodeAmplitudes),width=1.0,color=:blue,strokecolor=:blue)
Label(fig[1,2,Bottom()],L"(b)",textsize = 32)

# psi_v potential axis
Lₜ = makeLt(conditionsDict["params"],matricesDict["matrices"],T,linkTriangleAreas,trapeziumAreas)
eigenvectors = (eigen(Matrix(Lₜ))).vectors
eigenvalues = (eigen(Matrix(Lₜ))).values
vertexDivs = -1.0.*calculateVertexDivs(conditionsDict["params"],matricesDict["matrices"],q,linkTriangleAreas)
onesVec = ones(nVerts)
E = Diagonal(linkTriangleAreas)
ḡ = ((onesVec'*E*vertexDivs)/(onesVec'*E*ones(nVerts))).*onesVec
ğ = vertexDivs.-ḡ
ψ̆ = zeros(nVerts)
eigenmodeAmplitudes = Float64[]
for k=2:nVerts
    numerator = -eigenvectors[:,k]'*E*ğ
    denominator = eigenvalues[k]*(eigenvectors[:,k]'*E*eigenvectors[:,k])
    ψ̆ .+= (numerator/denominator).*eigenvectors[:,k]
    push!(eigenmodeAmplitudes,(numerator/denominator))
end
ψ̆Lims = (-maximum(abs.(ψ̆)),maximum(abs.(ψ̆)))
ax3 = Axis(fig[2,1],aspect=DataAspect())
hidedecorations!(ax3)
hidespines!(ax3)
for k=1:nVerts
    poly!(ax3,linkTriangles[k],color=[ψ̆[k]],colorrange=ψ̆Lims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25)) #:bwr
end
for i=1:nCells
    poly!(ax3,cellPolygons[i],color=(:white,0.0),strokecolor=(:black,1.0),strokewidth=1) #:bwr
end
# Colorbar(fig[1,1][1,2],limits=ψ̆Lims,colormap=:bwr,flipaxis=false,align=:left)
Label(fig[2,1,Bottom()],L"(c)",textsize = 32)

ax4 = Axis(fig[2,2], xlabel="Eigenmode number, k", ylabel="Amplitude", alignmode = Outside())
xlims!(ax4,0,nVerts)
ylims!(ax4,0,1.1*maximum(abs.(eigenmodeAmplitudes)))
barplot!(ax4,collect(2:nVerts),abs.(eigenmodeAmplitudes),width=1.0,color=:orange,strokecolor=:orange)
Label(fig[2,2,Bottom()],L"(d)",textsize = 32)

# Capital psi_v potential axis
Lₜ = makeLt(conditionsDict["params"],matricesDict["matrices"],T,linkTriangleAreas,trapeziumAreas)
eigenvectors = (eigen(Matrix(Lₜ))).vectors
eigenvalues = (eigen(Matrix(Lₜ))).values
vertexCurls = calculateVertexCurls(conditionsDict["params"],matricesDict["matrices"],q,linkTriangleAreas)
onesVec = ones(nVerts)
E = Diagonal(linkTriangleAreas)
ḡ = ((onesVec'*E*vertexCurls)/(onesVec'*E*ones(nVerts))).*onesVec
ğ = vertexCurls.-ḡ
ψ̆ = zeros(nVerts)
eigenmodeAmplitudes = Float64[]
for k=2:nVerts
    numerator = -eigenvectors[:,k]'*E*ğ
    denominator = eigenvalues[k]*(eigenvectors[:,k]'*E*eigenvectors[:,k])
    ψ̆ .+= (numerator/denominator).*eigenvectors[:,k]
    push!(eigenmodeAmplitudes,(numerator/denominator))
end
ψ̆Lims = (-maximum(abs.(ψ̆)),maximum(abs.(ψ̆)))
ax5 = Axis(fig[3,1],aspect=DataAspect())
hidedecorations!(ax5)
hidespines!(ax5)
for k=1:nVerts
    poly!(ax5,linkTriangles[k],color=[ψ̆[k]],colorrange=ψ̆Lims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25)) #:bwr
end
for i=1:nCells
    poly!(ax5,cellPolygons[i],color=(:white,0.0),strokecolor=(:black,1.0),strokewidth=1) #:bwr
end
# Colorbar(fig[1,1][1,2],limits=ψ̆Lims,colormap=:bwr,flipaxis=false,align=:left)
Label(fig[3,1,Bottom()],L"(e)",textsize = 32)

# Capital psi_v spectrum axis
ax6 = Axis(fig[3,2], xlabel="Eigenmode number, k", ylabel="Amplitude", alignmode = Outside())
xlims!(ax6,0,nVerts)
ylims!(ax6,0,1.1*maximum(abs.(eigenmodeAmplitudes)))
barplot!(ax6,collect(2:nVerts),abs.(eigenmodeAmplitudes),width=1.0,color=:green,strokecolor=:green)
Label(fig[3,2,Bottom()],L"(f)",textsize = 32)