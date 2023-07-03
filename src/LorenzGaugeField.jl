
struct LorenzGaugeField{T, U} <: AbstractLorenzGaugeField
  imex::T
  ρJs⁰::OffsetArray{Float64, 4, Array{Float64, 4}}
  ϕ⁺::Array{ComplexF64, 2}
  ϕ⁰::Array{ComplexF64, 2}
  Ax⁺::Array{ComplexF64, 2}
  Ay⁺::Array{ComplexF64, 2}
  Az⁺::Array{ComplexF64, 2}
  Ax⁰::Array{ComplexF64, 2}
  Ay⁰::Array{ComplexF64, 2}
  Az⁰::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Ez::Array{ComplexF64, 2}
  ρ⁰::Array{ComplexF64, 2}
  Jx⁰::Array{ComplexF64, 2}
  Jy⁰::Array{ComplexF64, 2}
  Jz⁰::Array{ComplexF64, 2}
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

function LorenzGaugeField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
    imex::AbstractImEx=Explicit(), buffer=0)
  EBxyz = OffsetArray(zeros(6, NX+2buffer, NY+2buffer), 1:6, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  ρJs = OffsetArray(zeros(4, NX+2buffer, NY+2buffer, nthreads()),
    1:4, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer, 1:nthreads());
  return LorenzGaugeField(imex, ρJs,
    (zeros(ComplexF64, NX, NY) for _ in 1:18)..., EBxyz, # 22
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end


function warmup!(field::LorenzGaugeField, plasma, to)
  ρcallback(a, b, c, d) = (a,)
  Jcallback(a, b, c, d) = (b, c, d)
  @timeit to "Warmup" begin
    dt = timestep(field)
    #field.ρjs⁰ .= 0
    #advect!(plasma, field.gridparams, -3dt/2, to) # n-3/2
    #deposit!(field.ρjs⁰, plasma, field.gridparams, dt, to, ρcallback)
    #advect!(plasma, field.gridparams, dt/2, to) # n -1
    #deposit!(field.ρjs⁰, plasma, field.gridparams, dt, to, jcallback)
    #reduction!(field.ρ⁰, field.jx⁰, field.jy⁰, field.jz⁰, field.ρjs⁰)
    #neglaplacesolve!(field.ϕ⁻, field.ρ⁰, field.ffthelper)
    #neglaplacesolve!(field.Ax⁻, field.Jx⁰, field.ffthelper)
    #neglaplacesolve!(field.Ay⁻, field.Jy⁰, field.ffthelper)
    #neglaplacesolve!(field.Az⁻, field.Jz⁰, field.ffthelper)

    #field.ρJs⁰ .= 0
    #advect!(plasma, field.gridparams, dt/2, to) # n-1/2
    #deposit!(field.ρJs⁰, plasma, field.gridparams, dt, to, ρcallback)
    #advect!(plasma, field.gridparams, dt/2, to) # n
    #deposit!(field.ρJs⁰, plasma, field.gridparams, dt, to, Jcallback)
    #reduction!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰)
    #neglaplacesolve!(field.ϕ⁰, field.ρ⁰, field.ffthelper)
    #neglaplacesolve!(field.Ax⁰, field.Jx⁰, field.ffthelper)
    #neglaplacesolve!(field.Ay⁰, field.Jy⁰, field.ffthelper)
    #neglaplacesolve!(field.Az⁰, field.Jz⁰, field.ffthelper)


    #advect!(plasma, field.gridparams, -dt/2, to) # n - 1/2
    #deposit!(field.ρJs⁰, plasma, field.gridparams, dt, to, ρcallback)
    #advect!(plasma, field.gridparams, dt/2, to) # n + 0
    #reduction!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰)

#    neglaplacesolve!(field.ϕ⁻, -field.ρ⁰, field.ffthelper)
  end
end


function loop!(plasma, field::LorenzGaugeField, to, t)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)
  # Assume ρ and J are up to date at the current time (n+0)
  # At this point Ai⁰ stores the (n+0)th timestep value and Ai⁻ the (n-1)th
  #               ϕ⁰  stores the (n-1/2)th timestep value and ϕ⁻ the (n-3/2)th
  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft! * field.ρ⁰;
    field.ffthelper.pfft! * field.Jx⁰;
    field.ffthelper.pfft! * field.Jy⁰;
    field.ffthelper.pfft! * field.Jz⁰;
    # smoothinfourierspace!(field.Jx⁰, field.ffthelper) # better for energy, worse for momentum
    # smoothinfourierspace!(field.Jy⁰, field.ffthelper) # better for energy, worse for momentum
    # smoothinfourierspace!(field.Jz⁰, field.ffthelper) # better for energy, worse for momentum
    # not necessary given E, B calculated from derivatives of ϕ, Ai
    #field.ρ⁰[1, 1] = field.Jx⁰[1, 1] = field.Jy⁰[1, 1] = field.Jz⁰[1, 1] = 0
  end

  @timeit to "Field Solve" begin
    chargeconservation!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.ffthelper, dt)
    lorenzgauge!(field.imex, field.ϕ⁺,  field.ϕ⁰,  field.ρ⁰,  field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ax⁺, field.Ax⁰, field.Jx⁰, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ay⁺, field.Ay⁰, field.Jy⁰, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Az⁺, field.Az⁰, field.Jz⁰, field.ffthelper.k², dt^2)
    #lorenzgauge!(field.imex, field.ϕ⁺,  field.ϕ⁰,  field.ϕ⁻,  field.ρ⁰,  field.ffthelper.k², dt^2)
    #lorenzgauge!(field.imex, field.Ax⁺, field.Ax⁰, field.Ax⁻, field.Jx⁰, field.ffthelper.k², dt^2)
    #lorenzgauge!(field.imex, field.Ay⁺, field.Ay⁰, field.Ay⁻, field.Jy⁰, field.ffthelper.k², dt^2)
    #lorenzgauge!(field.imex, field.Az⁺, field.Az⁰, field.Az⁻, field.Jz⁰, field.ffthelper.k², dt^2)
  end

  # at this point (ϕ, Ai) stores the (n+1)th timestep value and (ϕ⁻, Ai⁻) the nth
  # Now calculate the value of E and B at n+1/2
  # Eʰ = -∇ϕ⁺ - (A⁺ - A⁰)/dt
  # Bʰ = ∇x(A⁺ + A⁰)/2
  #
  #  ...E.....E... E⁰'⁵ = -∇ ϕ⁰'⁵ - (A¹ - A⁰) / Δt
  #  ...B.....B... B⁰'⁵ = ∇ x (A¹ + A⁰) / 2
  #  ...ϕ.....ϕ... ∂ₜ^2 ϕ⁰'⁵ = ∇² ϕ + (ρ⁻⁰'⁵ - ∇⋅J)
  #  A..0..A..+..A ∂ₜ^2 A = ∇² A + J
  #  ...ρ.....ρ... ∂ₜρ = -∇⋅J
  #  J.....J.....J J¹ = q n(x¹) v¹
  #  v.....v.....v v¹ = v⁰ + q Δt (E⁰'⁵ + v x B⁰'⁵) / m
  #  x.....x.....x x¹ = x⁰ + Δt (v¹ + v⁰) / 2
  @timeit to "Calculate E, B" begin
    @. field.Ex = -im * field.ffthelper.kx * field.ϕ⁺
    @. field.Ey = -im * field.ffthelper.ky * field.ϕ⁺
    @. field.Ex -= (field.Ax⁺ - field.Ax⁰)/dt
    @. field.Ey -= (field.Ay⁺ - field.Ay⁰)/dt
    @. field.Ez = -(field.Az⁺ - field.Az⁰)/dt
    @. field.Bx =  im * field.ffthelper.ky * (field.Az⁺ + field.Az⁰)/2
    @. field.By = -im * field.ffthelper.kx * (field.Az⁺ + field.Az⁰)/2
    @. field.Bz =  im * field.ffthelper.kx * (field.Ay⁺ + field.Ay⁰)/2
    @. field.Bz -= im * field.ffthelper.ky * (field.Ax⁺ + field.Ax⁰)/2
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
    @threads for j in axes(field.ρJs⁰, 4)
      ρJ⁰ = @view field.ρJs⁰[:, :, :, j]
      ρJ⁰ .= 0
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
          Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
          vxi, vyi = vx[i], vy[i]
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, q_m);
            x[i] = unimod(x[i] + (vxi + vx[i]) * dt / 2, Lx)
            y[i] = unimod(y[i] + (vyi + vy[i]) * dt / 2, Ly)
          deposit!(ρJ⁰, species.shape, x[i], y[i], NX_Lx, NY_Ly,
            vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
    reduction!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰)
  end

  #@timeit to "Copy over buffers" begin
  #  field.ϕ⁻ .= field.ϕ⁰
  #  field.ϕ⁰ .= field.ϕ⁺
  #  field.Ax⁻ .= field.Ax⁰
  #  field.Ax⁰ .= field.Ax⁺
  #  field.Ay⁻ .= field.Ay⁰
  #  field.Ay⁰ .= field.Ay⁺
  #  field.Az⁻ .= field.Az⁰
  #  field.Az⁰ .= field.Az⁺
  #end
end

