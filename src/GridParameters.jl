struct GridParameters
  Lx::Float64
  Ly::Float64
  NX::Int
  NY::Int
  ΔX::Float64
  ΔY::Float64
  NX_Lx::Float64
  NY_Ly::Float64
  xs::Vector{Float64}
  ys::LinearAlgebra.Adjoint{Float64, Vector{Float64}}
end
function GridParameters(Lx, Ly, NX, NY)
  xs = collect(Lx .* (0.5:NX));
  ys = collect(Ly .* (0.5:NY))';
  return GridParameters(Lx, Ly, NX, NY, Lx/NX, Ly/NY, NX / Lx, NY / Ly, xs, ys)
end

cellvolume(g::GridParameters) = g.ΔX * g.ΔY


