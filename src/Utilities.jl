unimod(x, n) = x > n ? x - n : x > 0 ? x : x + n

function applyperiodicity!(a::Array, oa)
  NX, NY = size(a)
  @assert length(size(a)) == 2
  @assert length(size(oa)) == 2
  for j in axes(oa, 2), i in axes(oa, 1)
    a[unimod(i, NX), unimod(j, NY)] += oa[i, j]
  end
end

function applyperiodicity!(oa, a::Array)
  NX, NY = size(a)
  @assert length(size(a)) == 2
  @assert length(size(oa)) == 2
  for j in axes(oa, 2), i in axes(oa, 1)
     oa[i, j] += real(a[unimod(i, NX), unimod(j, NY)])
  end
end

function reduction!(a, z)
  @. a = 0.0
  @views for k in axes(z, 3)
    applyperiodicity!(a, z[:, :, k])
  end
end

function reduction!(a, b, c, z)
  @assert size(z, 1) == 4
  @. a = 0.0
  @. b = 0.0
  @. c = 0.0
  @views for k in axes(z, 4)
    applyperiodicity!(a, z[1, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(b, z[2, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(c, z[3, :, :, k])
  end
end

function reduction!(a, b, c, d, z)
  @assert size(z, 1) == 4
  @. a = 0.0
  @. b = 0.0
  @. c = 0.0
  @. d = 0.0
  @views for k in axes(z, 4)
    applyperiodicity!(a, z[1, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(b, z[2, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(c, z[3, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(d, z[4, :, :, k])
  end
end

warmup!(field::AbstractField, plasma, to) = field

function warmup!(ρ, Jx, Jy, Jz, ρJs, plasma, gridparams, dt, to)
  Lx, Ly = gridparams.Lx, gridparams.Ly
  NX_Lx, NY_Ly = gridparams.NX_Lx, gridparams.NY_Ly
  ΔV = cellvolume(gridparams)
  @timeit to "Particle Loop" begin
    @threads for j in axes(ρJs, 4)
      ρJ = @view ρJs[:, :, :, j]
      ρJ .= 0
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        for i in species.chunks[j]
          x[i] = unimod(x[i] + vx[i] * dt, Lx)
          y[i] = unimod(y[i] + vy[i] * dt, Ly)
          deposit!(ρJ, species.shape, x[i], y[i], NX_Lx, NY_Ly,
            qw_ΔV, vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
          x[i] = unimod(x[i] - vx[i] * dt, Lx)
          y[i] = unimod(y[i] - vy[i] * dt, Ly)
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
    reduction!(ρ, Jx, Jy, Jz, ρJs)
  end
end
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
  ρₘ = sum(s->numberdensity(s, Lx * Ly) * s.mass, plasma)
  Va = B0 / sqrt(ρₘ)
  println("Resolution information:")
  println("    Va / c ", Va)
  for (s, species) in enumerate(plasma)
    println("  Species $s with mass $(species.mass) and charge $(species.charge):")
    Ω = cyclotronfrequency(species, B0)
    vrms = rmsvelocity(species)
    Π = plasmafrequency(species, Lx * Ly)
    λ_D = vrms / Π
    r_L = vrms / Ω
    println("    vrms / c ", vrms)
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

