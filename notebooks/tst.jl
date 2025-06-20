# Lf => Primary network, cells
# cocurlᶜ => Primary network, cells
# curlᶜ => Primary network, cells
# Lv => Primary network, vertices
# divᵛ => Primary network, vertices
# codᵛ => Primary network, vertices
# Lc => Dual network, cells 
# divᶜ => Dual network, cells 
# codᶜ => Dual network, cells 
# Lt => Dual network, vertices 
# cocurlᵛ => Dual network, vertices 
# curlᵛ => Dual network, vertices 


# Primary network 

# Old: Lf ψᶜ = -divᶜ 𝐡 
# Conversion: -divᶜ => cocurlᶜ
# New: Lf ψᶜ = cocurlᶜ 𝐡 

# Old: Lf Ψᶜ = curlᶜ 𝐡
# Conversion: curlᶜ => -curlᶜ
# New: Lf Ψᶜ = -curlᶜ 𝐡

# Old: Lᵥ ψᵛ = -d̃ivᵛ 𝐡
# Conversion: -d̃ivᵛ => -divᵛ
# New: Lᵥ ψᵛ = -divᵛ 𝐡

# Old: Lᵥ Ψᵛ = c̃urlᵛ 𝐡
# Conversion: c̃urlᵛ => codᵛ 
# New: Lᵥ Ψᵛ = codᵛ 𝐡


# Dual network 

#  Old: Lc ψ̆ᶜ = -d̃ivᶜ h̆
#  Conversion: -d̃ivᶜ => -divᶜ
#  New: Lc ψ̆ᶜ = -divᶜ h̆

#  Old: Lc Ψ̆ᶜ = C̃URLᶜ h̆
#  Conversion: C̃URLᶜ => codᶜ
#  New: Lc Ψ̆ᶜ = codᶜ h̆

#  Old: Lt ψ̆ᵛ = -divᵛ h̆
#  Conversion: -divᵛ => cocurlᵛ
#  New: Lt ψ̆ᵛ = cocurlᵛ h̆

#  Old: Lt Ψ̆ᵛ = CURLᵛ h̆
#  Conversion: CURLᵛ => -curlᵛ
#  New: Lt Ψ̆ᵛ = -curlᵛ h̆



# cocurlᵛ corotᵛ curlᵛ rotᵛ


# cocurlᶜ => Primary network, cells
# curlᶜ => Primary network, cells
# divᵛ => Primary network, vertices
# codᵛ => Primary network, vertices
# divᶜ => Dual network, cells 
# codᶜ => Dual network, cells 
# cocurlᵛ => Dual network, vertices 
# curlᵛ => Dual network, vertices 

# Lf => Primary network, cells
# cocurlᶜ => Primary network, cells
# curlᶜ => Primary network, cells
# Lv => Primary network, vertices
# divᵛ => Primary network, vertices
# codᵛ => Primary network, vertices
# Lc => Dual network, cells 
# divᶜ => Dual network, cells 
# codᶜ => Dual network, cells 
# Lt => Dual network, vertices 
# cocurlᵛ => Dual network, vertices 
# curlᵛ => Dual network, vertices 


# Primary network 

# Old: Lf ψᶜ = -divᶜ 𝐡 
# Conversion: -divᶜ => cocurlᶜ
# New: Lf ψᶜ = cocurlᶜ 𝐡 

# Old: Lf Ψᶜ = curlᶜ 𝐡
# Conversion: curlᶜ => -curlᶜ
# New: Lf Ψᶜ = -curlᶜ 𝐡

# Old: Lᵥ ψᵛ = -d̃ivᵛ 𝐡
# Conversion: -d̃ivᵛ => -divᵛ
# New: Lᵥ ψᵛ = -divᵛ 𝐡

# Old: Lᵥ Ψᵛ = c̃urlᵛ 𝐡
# Conversion: c̃urlᵛ => codᵛ 
# New: Lᵥ Ψᵛ = codᵛ 𝐡


# Dual network 

#  Old: Lc ψ̆ᶜ = -d̃ivᶜ h̆
#  Conversion: -d̃ivᶜ => -divᶜ
#  New: Lc ψ̆ᶜ = -divᶜ h̆

#  Old: Lc Ψ̆ᶜ = C̃URLᶜ h̆
#  Conversion: C̃URLᶜ => codᶜ
#  New: Lc Ψ̆ᶜ = codᶜ h̆

#  Old: Lt ψ̆ᵛ = -divᵛ h̆
#  Conversion: -divᵛ => cocurlᵛ
#  New: Lt ψ̆ᵛ = cocurlᵛ h̆

#  Old: Lt Ψ̆ᵛ = CURLᵛ h̆
#  Conversion: CURLᵛ => -curlᵛ
#  New: Lt Ψ̆ᵛ = -curlᵛ h̆




# function makeLf(nCells,B,Bᵀ,cellAreas,edgeLengths,trapeziumAreas)
#     onesVec = ones(1,nCells)
#     boundaryEdges = abs.(onesVec*B)
#     H = Diagonal(cellAreas)
#     boundaryEdgesFactor = abs.(boundaryEdges.-1)# =1 for internal vertices, =0 for boundary vertices
#     diagonalComponent = (boundaryEdgesFactor'.*((edgeLengths.^2)./(2.0.*trapeziumAreas)))[:,1] # Multiply by boundaryEdgesFactor vector to set boundary vertex contributions to zero
#     Tₑ = Diagonal(diagonalComponent)
#     invH = inv(H)
#     Lf = invH*B*Tₑ*Bᵀ
#     dropzeros!(Lf)    
#     return Lf
# end

# function makeLc(nCells,B,Bᵀ,cellAreas,T,trapeziumAreas)
#     onesVec = ones(1,nCells)
#     boundaryEdges = abs.(onesVec*B)
#     boundaryEdgesFactor = abs.(boundaryEdges.-1)# =1 for internal vertices, =0 for boundary vertices
#     H = Diagonal(cellAreas)
#     Tₗ = Diagonal(((norm.(T)).^2)./(2.0.*trapeziumAreas))
#     invTₗ = inv(Tₗ)
#     boundaryEdgesFactorMat = Diagonal(@view boundaryEdgesFactor[1,:])
#     Lc = (H\B)*boundaryEdgesFactorMat*invTₗ*Bᵀ
#     dropzeros!(Lc)
#     return Lc
# end

# function makeLv(A,Aᵀ,edgeLengths,linkTriangleAreas,trapeziumAreas)
#     E = Diagonal(linkTriangleAreas)
#     Tₑ = Diagonal((edgeLengths.^2)./(2.0.*trapeziumAreas))
#     Lᵥ = (E\Aᵀ)*(Tₑ\A)
#     dropzeros!(Lᵥ)
#     return Lᵥ
# end

# function makeLt(A,Aᵀ,T,linkTriangleAreas,trapeziumAreas)
#     E = Diagonal(linkTriangleAreas)
#     Tₗ = Diagonal(((norm.(T)).^2)./(2.0.*trapeziumAreas))
#     Lₜ = (E\Aᵀ)*Tₗ*A
#     dropzeros!(Lₜ)
#     return Lₜ
# end
