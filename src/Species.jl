
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

nmacroparticles(s::Species) = length(s.p)
thermalvelocity(s::Species) = std(velocities(s))
kineticenergy(s::Species) = sum(abs2, velocities(s)) * s.mass / 2 * s.weight
kineticenergydensity(s::Species, volume) = kineticenergy(s) / volume
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
momentumdensity(s::Species, volume) = momentum(s) / volume
function currentdensity(s::Species, volume)
  return momentumdensity(s, volume) / s.mass * s.charge
end

characteristicmomentumdensity(s::Species, volume) = momentum(s, abs) / volume

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

function bfieldvrotationmatrix(bvector)
  zvec = [0, 0, 1]
  B0 = norm(bvector)
  iszero(B0) && (bvector .= zvec)
  normvector = bvector ./ norm(bvector)
  θ = acos(dot(normvector, zvec))
  ux, uy, uz = cross(zvec, normvector)

  r00 = cos(θ) + ux^2 * (1 - cos(θ))
  r01 = ux * uy * (1 - cos(θ)) - uz * sin(θ)
  r02 = ux * uz * (1 - cos(θ)) + uy * sin(θ)
  r10 = uy * ux * (1 - cos(θ)) + uz * sin(θ)
  r11 = cos(θ) + uy^2 * (1 - cos(θ))
  r12 = uy * uz * (1 - cos(θ)) - ux * sin(θ)
  r20 = uz * ux * (1 - cos(θ)) - uy * sin(θ)
  r21 = uz * uy * (1 - cos(θ)) + ux * sin(θ)
  r22 = cos(θ) + uz^2 * (1 - cos(θ))
  R = [r00 r01 r02; r10 r11 r12; r20 r21 r22]
  return R
end

function ringbeaminitialiser(P, vth, mass, v0, bvector, pitch)
  rvparas = erfinv.(2sample(P, 5, 0.0) .- 1);
  rvparas .-= mean(rvparas)
  vpara = vth .* rvparas .+ v0 * pitch;
  vdriftperp = v0 * sqrt(1 - pitch^2);
  vperps = zeros(P)
  vperp_min = max(0.0, vdriftperp - 6.0 * vth)
  vperp_peak = (vdriftperp + sqrt(vdriftperp^2 + 2vth^2)) / 2;
  # v exp(-(v-u)^2/vth^2)
  if vth > 0
    for i in 1:P
      while true
        vperp = vperp_min + rand() * 2.0 * 6.0 * vth
        vf_eval = vperp / vperp_peak * exp(-((vperp - vdriftperp)^2 / vth^2));
        @assert vf_eval <= 1 "Error in the accept reject algorithm, f > 1"
        if (rand() < vf_eval)
          vperps[i] = vperp
          break
        end
      end
    end
  end
  gyroangle = (1/P/2:1/P:1-1/P/2) .* 2π
  vperp1 = vperps .* cos.(gyroangle)
  vperp2 = vperps .* sin.(gyroangle)
  R = bfieldvrotationmatrix(bvector)
  vx = vperp1 .* R[1, 1] .+ vperp2 .* R[1, 2] .+ vpara .* R[1, 3]
  vy = vperp1 .* R[2, 1] .+ vperp2 .* R[2, 2] .+ vpara .* R[2, 3]
  vz = vperp1 .* R[3, 1] .+ vperp2 .* R[3, 2] .+ vpara .* R[3, 3]
  return (vx, vy, vz)
end

function momentumtovelocity!(pvx, pvy, pvz, mass)
  # now convert to velocity
  @threads for i in eachindex(pvx, pvy, pvz)
    γ = sqrt(1 + (pvx[i]^2 + pvy[i]^2 + pvz[i]^2) / mass^2)
    pvx[i] /= (mass * γ)
    pvy[i] /= (mass * γ)
    pvz[i] /= (mass * γ)
    all(pvx[i]^2 + pvy[i]^2 + pvz[i]^2 <= 1)
  end
  return (pvx, pvy, pvz)
end

function thermalinitialiser(P, vth, mass, _...)
  pvx = erfinv.(2sample(P, 5, 0.0) .- 1);
  pvy = erfinv.(2sample(P, 7, 0.0) .- 1);
  pvz = erfinv.(2sample(P, 9, 0.0) .- 1);
  pvx .-= mean(pvx)
  pvy .-= mean(pvy)
  pvz .-= mean(pvz)
  pvx .*= mass * (vth / sqrt(2)) / std(pvx);
  pvy .*= mass * (vth / sqrt(2)) / std(pvy);
  pvz .*= mass * (vth / sqrt(2)) / std(pvz);
  pvx .-= mean(pvx)
  pvy .-= mean(pvy)
  pvz .-= mean(pvz)
  momentumtovelocity!(pvx, pvy, pvz, mass)
end

function Species(P, vth, density, shape::AbstractShape;
    Lx, Ly, charge, mass, bfield=[0, 0, 1],
    velocityinitialiser::F=()->thermalinitialiser(P, vth, mass)) where F
  x = Lx * sample(P, 2, 0.0);
  y = Ly * sample(P, 3, 0.0);
  # us pvi to first mean momentum and then velocity, to save RAM
  vx, vy, vz = velocityinitialiser()
  p = collect(1:P)
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


