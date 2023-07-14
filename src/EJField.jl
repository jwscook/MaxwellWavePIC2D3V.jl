
struct EJField{T, U} <: AbstractLorenzGaugeField
  imex::T
  depositionbuffer::OffsetArray{Float64, 4, Array{Float64, 4}}
  ϕ⁰::Array{ComplexF64, 2}
  Ax⁰::Array{ComplexF64, 2}
  Ay⁰::Array{ComplexF64, 2}
  Az⁰::Array{ComplexF64, 2}
  Ax⁺::Array{ComplexF64, 2}
  Ay⁺::Array{ComplexF64, 2}
  Az⁺::Array{ComplexF64, 2}
  Ex⁰::Array{ComplexF64, 2}
  Ey⁰::Array{ComplexF64, 2}
  Ez⁰::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Ez::Array{ComplexF64, 2}
  Ex⁻::Array{ComplexF64, 2}
  Ey⁻::Array{ComplexF64, 2}
  Ez⁻::Array{ComplexF64, 2}
  ρ⁰::Array{ComplexF64, 2}
  ρ⁺::Array{ComplexF64, 2}
  Jx⁰::Array{ComplexF64, 2}
  Jy⁰::Array{ComplexF64, 2}
  Jz⁰::Array{ComplexF64, 2}
  Jx⁺::Array{ComplexF64, 2}
  Jy⁺::Array{ComplexF64, 2}
  Jz⁺::Array{ComplexF64, 2}
  Bx::Array{ComplexF64, 2}
  By::Array{ComplexF64, 2}
  Bz::Array{ComplexF64, 2}
  EBxyz::OffsetArray{Float64, 3, Array{Float64, 3}}
  Axyz::OffsetArray{Float64, 3, Array{Float64, 3}}
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::U
  boris::ElectromagneticBoris
  dt::Float64
end

function EJField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
        imex::AbstractImEx=Explicit(), buffers=(0, 0))
  buffers = length(buffers) == 1 ? (buffers, buffers) : buffers
  @assert length(buffers) == 2
  bufferx, buffery = buffers
  EBxyz = OffsetArray(zeros(6, NX+2bufferx, NY+2buffery), 1:6, -(bufferx-1):NX+bufferx, -(buffery-1):NY+buffery);
  Axyz = OffsetArray(zeros(3, NX+2bufferx, NY+2buffery), 1:3, -(bufferx-1):NX+bufferx, -(buffery-1):NY+buffery);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  depositarray = OffsetArray(zeros(4, NX+2bufferx, NY+2buffery, nthreads()),
    1:4, -(bufferx-1):NX+bufferx, -(buffery-1):NY+buffery, 1:nthreads());
  return EJField(imex, depositarray,
    (zeros(ComplexF64, NX, NY) for _ in 1:27)..., EBxyz, Axyz,# 22
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end

function warmup!(field::EJField, plasma, to)
  return nothing
end

function loop!(plasma, field::EJField, to, t)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  NX, NY = field.gridparams.NX, field.gridparams.NY
  ΔV = cellvolume(field.gridparams)

  # Assume ρ and J are up to date at the current time (n+0)
  # At this point Ai⁰ stores the (n+0)th timestep value and Ai⁻ the (n-1)th
  #               ϕ⁰  stores the (n-1/2)th timestep value and ϕ⁻ the (n-3/2)th
  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft! * field.Jx⁺;
    field.ffthelper.pfft! * field.Jy⁺;
    field.ffthelper.pfft! * field.Jz⁺;
    field.ffthelper.pfft! * field.Bx
    field.ffthelper.pfft! * field.By
    field.ffthelper.pfft! * field.Bz
  end

  # ∂²E/∂t² = - ∂J/∂t - ∇ × (∇ × E)
  # ∂²E/∂t² = - ∂J/∂t - ∇(∇⋅E) + ∇^2 E
  # ∂²E/∂t² = - ∂J/∂t - ∇ ρ + ∇^2 E
  # ∂²E/∂t² = - ∂J/∂t - ik ρ - i k^2 E
  # ∂B/∂t = -∇×E;  ∂/∂t ∇×A = -∇×E; ∂/∂t A = -E
  @timeit to "Calculate E, B" begin
    f = field
    θ = theta(field.imex)
    @. f.ρ⁺ = f.ρ⁰ - dt * im * (f.ffthelper.kx * f.Jx⁰ + f.ffthelper.ky * f.Jy⁰)
    # explicit
    #@. f.Ex=2f.Ex⁰-f.Ex⁻-dt^2*((f.Jx⁺-f.Jx⁰)/dt+im*f.ffthelper.kx*f.ρ⁰+f.ffthelper.k²*f.Ex)
    #@. f.Ey=2f.Ey⁰-f.Ey⁻-dt^2*((f.Jy⁺-f.Jy⁰)/dt+im*f.ffthelper.ky*f.ρ⁰+f.ffthelper.k²*f.Ey)
    #@. f.Ez=2f.Ez⁰-f.Ez⁻-dt^2*((f.Jz⁺-f.Jz⁰)/dt+0                     +f.ffthelper.k²*f.Ez)
    # implicit
    @. f.Ex=(2f.Ex⁰-f.Ex⁻-dt^2*((f.Jx⁺-f.Jx⁰)/dt+im*f.ffthelper.kx*(f.ρ⁺+f.ρ⁰)/2+f.ffthelper.k²*((1-θ)*f.Ex⁰+θ/2*f.Ex⁻)))/(1+dt^2*θ/2*f.ffthelper.k²)
    @. f.Ey=(2f.Ey⁰-f.Ey⁻-dt^2*((f.Jy⁺-f.Jy⁰)/dt+im*f.ffthelper.ky*(f.ρ⁺+f.ρ⁰)/2+f.ffthelper.k²*((1-θ)*f.Ey⁰+θ/2*f.Ey⁻)))/(1+dt^2*θ/2*f.ffthelper.k²)
    @. f.Ez=(2f.Ez⁰-f.Ez⁻-dt^2*((f.Jz⁺-f.Jz⁰)/dt+0                              +f.ffthelper.k²*((1-θ)*f.Ez⁰+θ/2*f.Ez⁻)))/(1+dt^2*θ/2*f.ffthelper.k²)
    f.Ex[1,1] = f.Ey[1,1] = f.Ez[1,1] = 0
    # either do this
    if true
      @. f.Bx = f.Bx - dt * im * f.ffthelper.ky * f.Ez
      @. f.By = f.By + dt * im * f.ffthelper.kx * f.Ez
      @. f.Bz = f.Bz + dt * im * (f.ffthelper.kx * f.Ey - f.ffthelper.ky * f.Ex)
      # or this, they are equivalent
    else
      @. f.ϕ⁰ = f.ρ⁺ / f.ffthelper.k² # ρ⁰
      f.ϕ⁰[1,1] = 0
      @. f.Ax⁺ = f.Ax⁰ - dt * ((f.Ex + f.Ex⁰) / 2 + im * f.ffthelper.kx * f.ϕ⁰)
      @. f.Ay⁺ = f.Ay⁰ - dt * ((f.Ey + f.Ey⁰) / 2 + im * f.ffthelper.ky * f.ϕ⁰)
      @. f.Az⁺ = f.Az⁰ - dt * ((f.Ez + f.Ez⁰) / 2)
      f.Ax⁺[1,1] = f.Ay⁺[1,1] = f.Az⁺[1,1] = 0
      @. f.Bx = im * (f.ffthelper.ky * (f.Az⁺ + f.Az⁰)) / 2
      @. f.By = im * (-f.ffthelper.kx * (f.Az⁺ + f.Az⁰)) / 2
      @. f.Bz = im * (f.ffthelper.kx * (f.Ay⁺ + f.Ay⁰) - f.ffthelper.ky * (f.Ax⁺ + f.Ax⁰))/ 2
    end
    f.Bx[1,1] = f.By[1,1] = f.Bz[1,1] = 0
  end

  @timeit to "Field Inverse FT" begin
    field.ffthelper.pifft! * field.Ex
    field.ffthelper.pifft! * field.Ey
    field.ffthelper.pifft! * field.Ez
    field.ffthelper.pifft! * field.Bx
    field.ffthelper.pifft! * field.By
    field.ffthelper.pifft! * field.Bz
  end

  @timeit to "Field Update" update!(field)
  # we now have the E and B fields at n+1/2

  @timeit to "Field Forward FT for buffers" begin
    field.ffthelper.pfft! * field.Ex
    field.ffthelper.pfft! * field.Ey
    field.ffthelper.pfft! * field.Ez
  end

  @timeit to "Copy over buffers" begin
    field.ρ⁰ .= field.ρ⁺
    field.Ex⁻ .= field.Ex⁰
    field.Ex⁰ .= field.Ex
    field.Ey⁻ .= field.Ey⁰
    field.Ey⁰ .= field.Ey
    field.Ez⁻ .= field.Ez⁰
    field.Ez⁰ .= field.Ez
    field.Jx⁰ .= field.Jx⁺
    field.Jy⁰ .= field.Jy⁺
    field.Jz⁰ .= field.Jz⁺
    field.Ax⁰ .= field.Ax⁺
    field.Ay⁰ .= field.Ay⁺
    field.Az⁰ .= field.Az⁺
  end

  @timeit to "Particle Loop" begin
    particlemomenta = [zeros(3) for _ in 1:size(field.depositionbuffer, 4)]
    @threads for j in axes(field.depositionbuffer, 4)
      J⁰ = @view field.depositionbuffer[2:4, :, :, j]
      J⁰ .= 0
      ρ⁰ = @view field.depositionbuffer[1, :, :, j]
      ρ⁰ .= 0
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        q_m = species.charge / species.mass
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        #  E.....E.....E
        #  B.....B.....B
        #  ...ϕ.....ϕ.....ϕ
        #  A..0..A..+..A
        #  ...ρ.....ρ.....ρ
        #  J.....J.....J
        #  x.....x.....x
        #  v.....v.....v
        @inbounds for i in species.chunks[j]
          Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shapes, x[i], y[i])
          vxi, vyi = vx[i], vy[i]
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, q_m);
          x[i] = unimod(x[i] + (vxi + vx[i]) * dt/4, Lx)
          y[i] = unimod(y[i] + (vxi + vy[i]) * dt/4, Ly)
          deposit!(J⁰, species.shapes, x[i], y[i], NX_Lx, NY_Ly,
            vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
          x[i] = unimod(x[i] + (vxi + vx[i]) * dt/4, Lx)
          y[i] = unimod(y[i] + (vxi + vy[i]) * dt/4, Ly)
          deposit!(ρ⁰, species.shapes, x[i], y[i], NX_Lx, NY_Ly,
                   qw_ΔV)
          particlemomenta[j] .+= (vx[i], vy[i], vz[i]) .* (species.mass * species.weight)
        end
      end
    end
  end

  @timeit to "Field Reduction" begin
    reduction!(field.ρ⁰, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.depositionbuffer)
  end

end
function preparefieldsft!(f::EJField)
  f.ffthelper.pifft! * f.Ax⁰
  f.ffthelper.pifft! * f.Ay⁰
  f.ffthelper.pifft! * f.Az⁰
  f.ffthelper.pifft! * f.ϕ⁰
  f.ffthelper.pifft! * f.ρ⁰
  f.ffthelper.pifft! * f.Jx⁰
  f.ffthelper.pifft! * f.Jy⁰
  f.ffthelper.pifft! * f.Jz⁰
  f.ffthelper.pifft! * f.Ex
  f.ffthelper.pifft! * f.Ey
  f.ffthelper.pifft! * f.Ez
end
function restorefieldsft!(f::EJField)
  f.ffthelper.pfft! * f.Ax⁰
  f.ffthelper.pfft! * f.Ay⁰
  f.ffthelper.pfft! * f.Az⁰
  f.ffthelper.pfft! * f.ϕ⁰
  f.ffthelper.pfft! * f.ρ⁰
  f.ffthelper.pfft! * f.Jx⁰
  f.ffthelper.pfft! * f.Jy⁰
  f.ffthelper.pfft! * f.Jz⁰
  f.ffthelper.pfft! * f.Ex
  f.ffthelper.pfft! * f.Ey
  f.ffthelper.pfft! * f.Ez
end

function updatemomentum!(f::EJField)
  f.Axyz .= 0.0
  t0 = @spawn applyperiodicity!((@view f.Axyz[1, :, :]), f.Ax⁺)
  t1 = @spawn applyperiodicity!((@view f.Axyz[2, :, :]), f.Ay⁺)
  t2 = @spawn applyperiodicity!((@view f.Axyz[3, :, :]), f.Az⁺)
  wait.((t0, t1, t2))
end

@inline function (f::EJField)(shapes, xi::T, yi::T, Axyz) where T
  NX, NY = f.gridparams.NX, f.gridparams.NY
  NX_Lx, NY_Ly = f.gridparams.NX_Lx, f.gridparams.NY_Ly
  U = promote_type(T, real(eltype(Axyz)))
  output = MVector{3, U}(0.0, 0.0, 0.0)
  for (j, wy) in depositindicesfractions(shapes[2], yi, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(shapes[1], xi, NX, NX_Lx)
      wxy = wx * wy
      @simd for h in 1:3
          output[h] += Axyz[h, i, j] * wxy
      end
    end
  end
  return output
end


