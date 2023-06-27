
struct Species{S<:AbstractShape}
  charge::Float64
  mass::Float64
  weight::Float64
  shape::S
  xyv::Matrix{Float64}
  p::Vector{Int}
  chunks::Vector{UnitRange{Int}}
  xyvwork::Matrix{Float64}
end
function positions(s::Species; work=false)
  return work ? (@view s.xyvwork[1:2, :]) : (@view s.xyv[1:2, :])
end
function velocities(s::Species; work=false)
  return work ? (@view s.xyvwork[3:5, :]) : (@view s.xyv[3:5, :])
end
xyvchunk(s::Species, i::Int) = @view s.xyv[:, s.chunks[i]]

function copyto!(dest::Species, src::Species)
  @tturbo dest.xyv .= src.xyv
  @tturbo dest.p .= src.p
  return dest
end

rmsvelocity(s::Species) = sqrt(mean(abs2, velocities(s)))
kineticenergy(s::Species) = sum(abs2, velocities(s)) * s.mass / 2 * s.weight
cyclotronfrequency(s::Species, B0) = s.charge * B0 / s.mass
numberdensity(s::Species, volume) = (s.weight / volume * length(s.p))
function plasmafrequency(s::Species, volume)
  return sqrt(s.charge^2 * numberdensity(s, volume) / s.mass)
end

function momentum(s::Species, op::F=identity) where F
  #output = sum(op.(velocities(s)), dims=2)[:] * s.mass * s.weight
  output = @MVector [0.0, 0.0, 0.0]
  for v in eachcol(velocities(s))
    output .+= op.(v)
  end
  output .*= s.mass * s.weight
  return output
end
characteristicmomentum(s::Species) = momentum(s, abs)

calculateweight(n0, P, Lx, Ly) = n0 * Lx * Ly / P;

function halton(i, base, seed=0.0)
  result, f = 0.0, 1.0
  while i > 0
    f = f / base;
    result += f * mod(i, base)
    i ÷= base;
  end
  return mod(result + seed, 1)
end

#sample(P, base, seed) = halton.(0:P-1, base, 1/sqrt(2));#
#sample(P, _, _) = unimod.(rand() .+ reshape(QuasiMonteCarlo.sample(P,1,GoldenSample()), P), 1)
sample(P, _, _) = rand(P)

function Species(P, vth, density, shape::AbstractShape; Lx, Ly, charge=1, mass=1)
  x  = Lx * sample(P, 2, 0.0);
  y  = Ly * sample(P, 3, 0.0);
  vx = erfinv.(2sample(P, 5, 0.0) .- 1);
  vy = erfinv.(2sample(P, 7, 0.0) .- 1);
  vz = erfinv.(2sample(P, 9, 0.0) .- 1);
  vx .-= mean(vx)
  vy .-= mean(vy)
  vz .-= mean(vz)
  vx .*= (vth / sqrt(2)) / std(vx);
  vy .*= (vth / sqrt(2)) / std(vy);
  vz .*= (vth / sqrt(2)) / std(vz);
  p  = collect(1:P)
  xyv = Matrix(hcat(x, y, vx, vy, vz)')
  chunks = collect(Iterators.partition(1:P, ceil(Int, P/nthreads())))
  weight = calculateweight(density, P, Lx, Ly)
  return Species(Float64(charge), Float64(mass), weight, shape, xyv, p, chunks, deepcopy(xyv))
end

function Base.sort!(s::Species, Δx, Δy)
  sortperm!(s.p, eachindex(s.p),
    by=i->(ceil(Int, s.xyv[1,i] / Δx), ceil(Int, s.xyv[2,i] / Δy), s.xyv[3,i]))
  s.xyv .= s.xyv[:, s.p]
  return nothing
end


