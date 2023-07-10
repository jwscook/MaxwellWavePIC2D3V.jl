unimod(x, n) = x > n ? x - n : x > 0 ? x : x + n
#unimod(x, n) = mod1(x, n)
#function unimod(x, n)
#  (1 <= x <= n) && return x
#  while x > n
#    x = x - n
#  end
#  while x <= 0
#    x = x+ n
#  end
#  x
#end

function applyperiodicity!(a::Array, oa)
  NX, NY = size(a)
  @assert length(size(a)) == 2
  @assert length(size(oa)) == 2
  # this can't be threaded because the mod operator may make a data race
  for j in axes(oa, 2), i in axes(oa, 1)
    a[unimod(i, NX), unimod(j, NY)] += oa[i, j]
  end
end

function applyperiodicity!(oa, a::Array)
  NX, NY = size(a)
  @assert length(size(a)) == 2
  @assert length(size(oa)) == 2
  # this one can be threaded because the mod operator is on the rhs
  for j in axes(oa, 2)
     for i in axes(oa, 1)
       oa[i, j] += real(a[unimod(i, NX), unimod(j, NY)])
    end
  end
end

function reduction!(a, z)
  @. a = 0.0
  @views for k in axes(z, 3)
    applyperiodicity!(a, z[:, :, k])
  end
end

function reduction!(a, b, c, z)
  @assert size(z, 1) == 3
  a .= 0.0
  @views for k in axes(z, 4)
    applyperiodicity!(a, z[1, :, :, k])
  end
  b .= 0.0
  @views for k in axes(z, 4)
    applyperiodicity!(b, z[2, :, :, k])
  end
  c .= 0.0
  @views for k in axes(z, 4)
    applyperiodicity!(c, z[3, :, :, k])
  end
end

function reduction!(args...)
  z = args[end]
  @assert size(z, 1) == length(args) - 1
  for (h, a) in enumerate(args[1:end-1])
    a .= 0.0
    @views for k in axes(z, 4)
      applyperiodicity!(a, z[h, :, :, k])
    end
  end
end


warmup!(field::AbstractField, plasma, to) = field

function advect!(plasma, gridparams, dt, to)
  Lx, Ly = gridparams.Lx, gridparams.Ly
  @timeit to "Advect Loop" begin
    for species in plasma
      @threads for j in eachindex(species.chunks)
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        for i in species.chunks[j]
          x[i] = unimod(x[i] + vx[i] * dt, Lx)
          y[i] = unimod(y[i] + vy[i] * dt, Ly)
        end
      end
    end
  end
end

function printresolutions(plasma, field, dt, NT, to)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔX, ΔY = field.gridparams.ΔX, field.gridparams.ΔY
  ΔV = cellvolume(field.gridparams)
  Δl = sqrt(ΔX^2 + ΔY^2)
  B0 = norm(field.B0)
  nm = sum(s->numberdensity(s, Lx * Ly) * s.mass, plasma)
  Va = B0 / sqrt(nm)
  println("Resolution information:")
  println("    NT ", NT)
  println("    dt ", dt)
  println("    dt * NT ", dt * NT)
  println("    Lx ", Lx)
  println("    Ly ", Ly)
  println("    NX ", Int(NX_Lx * Lx))
  println("    NY ", Int(NY_Ly * Ly))
  println("    ΔX ", ΔX)
  println("    ΔY ", ΔY)
  println("    B0 ", B0)
  println("    Va / c ", Va)
  println("    c dt / ΔX ", dt / ΔX)
  println("    ΔX / (c dt) ", ΔX / dt)
  println("    c dt / ΔY ", dt / ΔY)
  println("    ΔY / (c dt) ", ΔY / dt)
  for (s, species) in enumerate(plasma)
    println("  Species $s with mass $(species.mass) and charge $(species.charge) and numberdensity $(nm):")
    Ω = cyclotronfrequency(species, B0)
    vth = thermalvelocity(species)
    Π = plasmafrequency(species, Lx * Ly)
    λ_D = vth / Π
    r_L = vth / Ω
    println("    vth / c ", vth)
    println("    vth / Va ", vth / Va)
    println("    c / vth ", 1 / vth)
    println("    vth dt / Δl ", vth * dt / Δl)
    println("    Δl / (vth dt) ", 1/(vth * dt / Δl))
    println("    λ_D / Δx ", λ_D / ΔX)
    println("    λ_D / Δy ", λ_D / ΔY)
    println("    r_L / Δl ", r_L / Δl)
    println("    (2π / Π) / Δt ", 2π/Π / dt)
    println("    (2π / Ω) / Δt ", 2π/Ω / dt)
    println("    T / (2π / Π) ", dt * NT / (2π/Π))
    println("    T / (2π / Ω) ", dt * NT / (2π/Ω))
    println("    (2π / Lx) Va / Ω ", (2π / Lx) * Va / Ω)
    println("    (2π / Ly) Va / Ω ", (2π / Ly) * Va / Ω)
  end
end

