module MaxwellWavePIC2D3V

using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Random
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings, MuladdMacro, CommonSubexpressions
using TimerOutputs, StaticNumbers, OffsetArrays, FastPow, ThreadsX
using QuasiMonteCarlo

abstract type AbstractShape end
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

include("ElectrostaticField.jl")
include("LorenzGaugeStaggeredField.jl")

include("Diagnostics.jl")
include("Solves.jl")
include("GridInteraction.jl")

end


