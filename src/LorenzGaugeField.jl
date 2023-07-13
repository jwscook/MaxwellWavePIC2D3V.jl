
struct LorenzGaugeField{T, U} <: AbstractLorenzGaugeField
  imex::T
  depositionbuffer::OffsetArray{Float64, 4, Array{Float64, 4}}
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
  px_q::Array{ComplexF64, 2}
  py_q::Array{ComplexF64, 2}
  pz_q::Array{ComplexF64, 2}
  ABExyz::OffsetArray{Float64, 3, Array{Float64, 3}}
  A⁺xyz::OffsetArray{Float64, 3, Array{Float64, 3}}
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::U
  boris::ElectromagneticBoris
  dt::Float64
end

function LorenzGaugeField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
        imex::AbstractImEx=Explicit(), buffers=(0,0))
  buffers = length(buffers) == 1 ? (buffers, buffers) : buffers
  @assert length(buffers) == 2
  bufferx, buffery = buffers
  ABExyz = OffsetArray(zeros(9, NX+2bufferx, NY+2buffery), 1:9, -(bufferx-1):NX+bufferx, -(buffery-1):NY+buffery);
  A⁺xyz = OffsetArray(zeros(3, NX+2bufferx, NY+2buffery), 1:3, -(bufferx-1):NX+bufferx, -(buffery-1):NY+buffery);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  depositionbuffer = OffsetArray(zeros(6, NX+2bufferx, NY+2buffery, nthreads()),
    1:6, -(bufferx-1):NX+bufferx, -(buffery-1):NY+buffery, 1:nthreads());
  return LorenzGaugeField(imex, depositionbuffer,
    (zeros(ComplexF64, NX, NY) for _ in 1:21)..., ABExyz, A⁺xyz,# 22
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end

function warmup!(field::LorenzGaugeField, plasma, to)
  return nothing
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
    #field.ffthelper.pfft! * field.ρ⁰; # field.ρ⁰ should always be in Fourier space
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
    # smoothinfourierspace!(field.ρ⁰, field.ffthelper)
    lorenzgauge!(field.imex, field.ϕ⁺,  field.ϕ⁰,  field.ρ⁰,  field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ax⁺, field.Ax⁰, field.Jx⁰, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ay⁺, field.Ay⁰, field.Jy⁰, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Az⁺, field.Az⁰, field.Jz⁰, field.ffthelper.k², dt^2)
    field.ϕ⁺[1, 1] = field.Ax⁺[1,1] = field.Ay⁺[1,1] = field.Az⁺[1,1] = 0 # just in case
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
    field.ffthelper.pifft! * field.Ax⁺
    field.ffthelper.pifft! * field.Ay⁺
    field.ffthelper.pifft! * field.Az⁺
    field.ffthelper.pifft! * field.Ax⁰
    field.ffthelper.pifft! * field.Ay⁰
    field.ffthelper.pifft! * field.Az⁰
  end

  @timeit to "Field Update" begin
    update!(field.ABExyz, (field.Ax⁰, field.Ay⁰, field.Az⁰,
                           field.Ex, field.Ey, field.Ez,
                           field.Bx, field.By, field.Bz))
    addB0!((@view field.ABExyz[7:9, :, :]), field.B0)
    update!(field.A⁺xyz, (field.Ax⁺, field.Ay⁺, field.Az⁺))
  end

  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft! * field.Ax⁺
    field.ffthelper.pfft! * field.Ay⁺
    field.ffthelper.pfft! * field.Az⁺
    field.ffthelper.pfft! * field.Ax⁰
    field.ffthelper.pfft! * field.Ay⁰
    field.ffthelper.pfft! * field.Az⁰
  end
  # we now have the E and B fields at n+1/2

  @timeit to "Particle Loop" begin
    @threads for j in axes(field.depositionbuffer, 4)
      three = @MVector zeros(3)
      nine = @MVector zeros(9)
      Jp_q = @view field.depositionbuffer[:, :, :, j]
      Jp_q .= 0
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        m_qΔV = species.mass / species.charge / ΔV * species.weight
        q_m = species.charge / species.mass
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        @inbounds for i in species.chunks[j]
          Axi, Ayi, Azi, Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(nine, species.shapes, field.ABExyz, x[i], y[i], @stat 9)
          @assert all(isfinite, (Exi, Eyi, Ezi, Bxi, Byi, Bzi))
          vxi, vyi, vzi = vx[i], vy[i], vz[i]
          γ = 1/sqrt(1 - vxi^2 - vyi^2 - vzi^2)
          pxi = vxi * species.mass * γ + Axi * species.charge * species.weight
          pyi = vyi * species.mass * γ + Ayi * species.charge * species.weight
          pzi = vzi * species.mass * γ + Azi * species.charge * species.weight
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, q_m);
          @assert all(isfinite, (x[i], y[i], vxi, vyi, vx[i], vy[i]))
          x[i] = unimod(x[i] + (vxi + vx[i]) * dt / 2, Lx)
          y[i] = unimod(y[i] + (vyi + vy[i]) * dt / 2, Ly)
          deposit!(Jp_q, species.shapes, x[i], y[i], NX_Lx, NY_Ly,
            # first 3 are current
            (vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV, # current
            # change in momentum per charge
            vx[i] * m_qΔV, vy[i] * m_qΔV, vz[i] * m_qΔV))
          Axi, Ayi, Azi = field(three, species.shapes, field.A⁺xyz, x[i], y[i], @stat 3)
          vnew = (vx[i], vy[i], vz[j])
          γ = 1/sqrt(1 - vxi^2 - vyi^2 - vzi^2)
          vx[i] = (pxi - Axi * species.charge) / species.mass / γ
          vy[i] = (pyi - Ayi * species.charge) / species.mass / γ
          vz[i] = (pzi - Azi * species.charge) / species.mass / γ
          vcorrected = (vx[i], vy[i], vz[j])
          #@show vcorrected ./ vnew
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
      reduction!(field.Jx⁰, field.Jy⁰, field.Jz⁰, (@view field.depositionbuffer[1:3, :, :, :]))
      reduction!(field.px_q, field.py_q, field.pz_q, (@view field.depositionbuffer[4:6, :, :, :]))
  end
#  field.ffthelper.pifft! * field.Ax⁺
#  @show mean(field.Ax⁺ ./ field.px_q)
#  @show std(field.Ax⁺ ./ field.px_q)
#  field.ffthelper.pfft! * field.Ax⁺
end

