

nPanels = 12
order = "PeffRising"
# order = "PeffFalling"
# order = "Radius"

# dateString = "25-12-02-16-58-52"
# dateString = "26-06-19-08-32-27"
# dateString = "26-06-19-08-32-27"
dateString = "26-07-07-16-49-05"
# dateString = "26-06-19-08-32-27"

function ϕ(θs, p)
    out = []
    for (i, θ) in enumerate(θs)
        tmp = p[1] + p[2]*cos(θ-p[3]) + p[4]*cos(2.0*(θ-p[5]))
        if abs(tmp) < 1.0
            push!(out, tmp)
        else
            push!(out, sign(tmp)*1.0)
        end
    end
    return out
end

function r(radii, p)
    out = []
    for (i, r) in enumerate(radii)
        tmp = p[1] + p[2]*r^p[3]
        push!(out, tmp)
    end
    return out
end

# Establish a set of cells for which ablated and divided results have been created
testFiles = filter(x->(occursin("Divided",x)||occursin("Ablated",x))&&occursin(".jld2",x), readdir(datadir("displacementFields", dateString)))
dividedCells = parse.(Int64, [i[(length(dateString)+1+length("Divided"))+1:end-5] for i in testFiles if occursin("Divided", i)])
ablatedCells = parse.(Int64, [i[(length(dateString)+1+length("Ablated"))+1:end-5] for i in testFiles if occursin("Ablated", i)])
# testCells = dividedCells∩ablatedCells

# Import and process data
importedData = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2");
                typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                            "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                )
            )
@unpack R, params, matrices = importedData

# Select which of the datasets to plot
cellCentres = findCellCentresOfMass(R, matrices.A, matrices.B)
𝐡 = hNetwork(R, matrices.A, matrices.B, matrices.F)
# Stress tensors 
σᵢ = σ(R, matrices.A, matrices.B, 𝐡)
# Deviatoric stress 
σDᵢ = [σᵢ[i] .- 0.5*tr(σᵢ[i]) for i=1:size(matrices.B,1)]
σDSᵢ = 0.5.*(σDᵢ .+ transpose.(σDᵢ))
# ~\ref{eq:shearstressexact}
ζᵢ = [sqrt(-det(σDSᵢ[i])) for i=1:size(matrices.B,1)]
Peffs_all = -0.5.*cocurlᶜ(R, matrices.A, matrices.B, 𝐡)
Peffs_ablated = Peffs_all[ablatedCells]
systemCOM = sum(R)./size(matrices.B,1)
radii_all = norm.([r.-systemCOM for r in cellCentres])
radii_ablated = radii_all[ablatedCells]

if order == "PeffRising"
    sortedOrder = sortperm(Peffs_ablated)
elseif order == "PeffFalling"    
    sortedOrder = sortperm(Peffs_ablated, rev=true)
elseif order=="Radius"     
    sortedOrder = sortperm(radii_ablated)
end