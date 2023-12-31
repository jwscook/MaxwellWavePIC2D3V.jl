module MaxwellWavePIC2D3V

using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Random
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings, MuladdMacro, CommonSubexpressions
using TimerOutputs, StaticNumbers, OffsetArrays, FastPow, ThreadsX
using QuasiMonteCarlo

abstract type AbstractShape end
Base.length(::AbstractShape) = 1
struct NGPWeighting <: AbstractShape end
struct AreaWeighting <: AbstractShape end
struct BSpline2Weighting <: AbstractShape end
struct BSplineWeighting{N} <: AbstractShape end

abstract type AbstractField end
abstract type AbstractLorenzGaugeField <: AbstractField end
timestep(f::AbstractField) = f.boris.dt_2 * 2

abstract type AbstractImEx end
struct Explicit <: AbstractImEx end
struct Implicit <: AbstractImEx end
struct ImEx <: AbstractImEx
  θ::Float64
end

theta(::Explicit) = 0
theta(imex::ImEx) = imex.θ
theta(::Implicit) = 1

include("Utilities.jl")
include("Boris.jl")
include("Species.jl")

include("GridParameters.jl")
include("FFTHelper.jl")


@inline function (f::AbstractLorenzGaugeField)(shapes, xi::T, yi::T) where T
  NX, NY = f.gridparams.NX, f.gridparams.NY
  NX_Lx, NY_Ly = f.gridparams.NX_Lx, f.gridparams.NY_Ly
  U = promote_type(T, real(eltype(f.EBxyz)))
  output = MVector{6, U}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for (j, wy) in depositindicesfractions(shapes[2], yi, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(shapes[1], xi, NX, NX_Lx)
      wxy = wx * wy
      @simd for h in 1:6
          output[h] += f.EBxyz[h, i, j] * wxy
      end
    end
  end
  return output
end

function update!(f::AbstractLorenzGaugeField)
  f.EBxyz .= 0.0
  t0 = @spawn applyperiodicity!((@view f.EBxyz[1, :, :]), f.Ex)
  t1 = @spawn applyperiodicity!((@view f.EBxyz[2, :, :]), f.Ey)
  t2 = @spawn applyperiodicity!((@view f.EBxyz[3, :, :]), f.Ez)
  t3 = @spawn applyperiodicity!((@view f.EBxyz[4, :, :]), f.Bx)
  t4 = @spawn applyperiodicity!((@view f.EBxyz[5, :, :]), f.By)
  t5 = @spawn applyperiodicity!((@view f.EBxyz[6, :, :]), f.Bz)
  wait.((t0, t1, t2, t3, t4, t5))
  @threads for k in axes(f.EBxyz, 3)
    for j in axes(f.EBxyz, 2), i in 1:3
      f.EBxyz[i+3, j, k] += @inbounds f.B0[i]
    end
  end
end


include("ElectrostaticField.jl")
include("LorenzGaugeField.jl")
include("LorenzGaugeSemiImplicitField.jl")
include("EJField.jl")

include("Diagnostics.jl")
include("Solves.jl")
include("GridInteraction.jl")

end


