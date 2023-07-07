
struct EJField{T, U} <: AbstractLorenzGaugeField
  imex::T
  Js⁰::OffsetArray{Float64, 4, Array{Float64, 4}}
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
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::U
  boris::ElectromagneticBoris
  dt::Float64
end

function EJField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
    imex::AbstractImEx=Explicit(), buffer=0)
  EBxyz = OffsetArray(zeros(6, NX+2buffer, NY+2buffer), 1:6, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  Js = OffsetArray(zeros(3, NX+2buffer, NY+2buffer, nthreads()),
    1:3, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer, 1:nthreads());
  return EJField(imex, Js,
    (zeros(ComplexF64, NX, NY) for _ in 1:26)..., EBxyz, # 22
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end

function warmup!(field::EJField, plasma, to)
  return nothing
end


function loop!(plasma, field::EJField, to, t)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)
  # Assume ρ and J are up to date at the current time (n+0)
  # At this point Ai⁰ stores the (n+0)th timestep value and Ai⁻ the (n-1)th
  #               ϕ⁰  stores the (n-1/2)th timestep value and ϕ⁻ the (n-3/2)th
  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft! * field.Jx⁺;
    field.ffthelper.pfft! * field.Jy⁺;
    field.ffthelper.pfft! * field.Jz⁺;
  end

  # ∂²E/∂t² = - ∂J/∂t - ∇ × (∇ × E)
  # ∂²E/∂t² = - ∂J/∂t - ∇(∇⋅E) + ∇^2 E
  # ∂²E/∂t² = - ∂J/∂t - ∇ ρ + ∇^2 E
  # ∂²E/∂t² = - ∂J/∂t - ik ρ - i k^2 E
  # ∂B/∂t = -∇×E;  ∂/∂t ∇×A = -∇×E; ∂/∂t A = -E
  @timeit to "Calculate E, B" begin
    f = field
    @. f.ρ⁰ = im * (f.ffthelper.kx * f.Ex + f.ffthelper.ky * f.Ey)
    @. f.Ex = 2f.Ex⁰ - f.Ex⁻ - dt^2 * ((f.Jx⁺ - f.Jx⁰) / dt - im * f.ffthelper.kx * f.ρ⁰ - f.ffthelper.k² * f.Ex)
    @. f.Ey = 2f.Ey⁰ - f.Ey⁻ - dt^2 * ((f.Jy⁺ - f.Jy⁰) / dt - im * f.ffthelper.ky * f.ρ⁰ - f.ffthelper.k² * f.Ey)
    @. f.Ez = 2f.Ez⁰ - f.Ex⁻ - dt^2 * ((f.Jz⁺ - f.Jz⁰) / dt - 0                          - f.ffthelper.k² * f.Ez)
    @. f.Ax⁺ = f.Ax⁰ - dt * f.Ex
    @. f.Ay⁺ = f.Ay⁰ - dt * f.Ey
    @. f.Az⁺ = f.Az⁰ - dt * f.Ez
    @. f.Bx =  im * f.ffthelper.ky * f.Az⁺
    @. f.By = -im * f.ffthelper.kx * f.Az⁺
    @. f.Bz =  im * f.ffthelper.kx * f.Ay⁺
    @. f.Bz -= im * f.ffthelper.ky * f.Ax⁺
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

  @timeit to "Particle Loop" begin
    @threads for j in axes(field.Js⁰, 4)
      J⁰ = @view field.Js⁰[:, :, :, j]
      J⁰ .= 0
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
          x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
          y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
          Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
#          @assert all(isfinite, (Exi, Eyi, Ezi, Bxi, Byi, Bzi))
#          vxi, vyi = vx[i], vy[i]
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, q_m);
          x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
          y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
#          @assert all(isfinite, (x[i], y[i], vxi, vyi, vx[i], vy[i]))
          deposit!(J⁰, species.shape, x[i], y[i], NX_Lx, NY_Ly,
            vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
        end
      end
    end
  end

  @timeit to "Field Inverse FT" begin
    field.ffthelper.pifft! * field.Ex
    field.ffthelper.pifft! * field.Ey
    field.ffthelper.pifft! * field.Ez
  end

  @timeit to "Copy over buffers" begin
    field.Ex⁻ .= field.Ex⁰
    field.Ex⁰ .= field.Ex
    field.Ey⁻ .= field.Ey⁰
    field.Ey⁰ .= field.Ey
    field.Ez⁻ .= field.Ez⁰
    field.Ez⁰ .= field.Ez
    field.Jx⁰ .= field.Jx⁺
    field.Jy⁰ .= field.Jy⁺
    field.Jz⁰ .= field.Jz⁺
  end
  @timeit to "Field Reduction" begin
    reduction!(field.Jx⁺, field.Jy⁺, field.Jz⁺, field.Js⁰)
  end
end

