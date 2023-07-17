
struct LorenzGaugeField{T, U} <: AbstractLorenzGaugeField
  imex::T
  Js⁰::OffsetArray{Float64, 4, Array{Float64, 4}}
  ϕ⁺::Array{ComplexF64, 2}
  ϕ⁰::Array{ComplexF64, 2}
  Ax⁺::Array{ComplexF64, 2}
  Ay⁺::Array{ComplexF64, 2}
  Az⁺::Array{ComplexF64, 2}
  Ax⁰::Array{ComplexF64, 2}
  Ay⁰::Array{ComplexF64, 2}
  Az⁰::Array{ComplexF64, 2}
  ρ⁰::Array{ComplexF64, 2}
  Jx⁰::Array{ComplexF64, 2}
  Jy⁰::Array{ComplexF64, 2}
  Jz⁰::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Ez::Array{ComplexF64, 2}
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
        imex::AbstractImEx=Explicit(), buffers=(0,0))
  buffers = length(buffers) == 1 ? (buffers, buffers) : buffers
  @assert length(buffers) == 2
  bufferx, buffery = buffers
  EBxyz = OffsetArray(zeros(6, NX+2bufferx, NY+2buffery), 1:6, -(bufferx-1):NX+bufferx, -(buffery-1):NY+buffery);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  Js = OffsetArray(zeros(3, NX+2bufferx, NY+2buffery, nthreads()),
    1:3, -(bufferx-1):NX+bufferx, -(buffery-1):NY+buffery, 1:nthreads());
  return LorenzGaugeField(imex, Js,
    (zeros(ComplexF64, NX, NY) for _ in 1:18)..., EBxyz, # 22
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end

function warmup!(field::LorenzGaugeField, plasma, to)
  return nothing
  @timeit to "Warmup" begin
    dt = timestep(field)
    Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
    NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
#    ΔV = cellvolume(field.gridparams)
#    advect!(plasma, field.gridparams, -dt, to) # n -1
    @threads for j in axes(field.Js⁰, 4)
      J⁰ = @view field.Js⁰[:, :, :, j]#[1, :, :, j]
      J⁰ .= 0
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        q_m = species.charge / species.mass
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        @inbounds for i in species.chunks[j]
          Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shapes, x[i], y[i])
          @assert all(isfinite, (Exi, Eyi, Ezi, Bxi, Byi, Bzi))
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, -0.5 * q_m); # minus sign here is to make it go backwards a dt/2
        end
      end
    end
#    #reduction!(field.ρ⁰, (@view field.Js⁰[1, :, :, :]))
#    reduction!(field.Jx⁰, field.Jy⁰, field.Jz⁰, field.Js⁰)
#    #neglaplacesolve!(field.ϕ⁰, field.ρ⁰, field.ffthelper)
#    advect!(plasma, field.gridparams, dt, to) # n
#    field.Js⁰ .= 0.0
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
    #field.ffthelper.pfft! * field.ρ⁰; # field.ρ⁰ should always be in Fourier space
    field.ffthelper.pfft! * field.Jx⁰;
    field.ffthelper.pfft! * field.Jy⁰;
    field.ffthelper.pfft! * field.Jz⁰;
    field.ffthelper.pfft! * field.Ex
    field.ffthelper.pfft! * field.Ey
    field.ffthelper.pfft! * field.Ez
    field.ffthelper.pfft! * field.Bx
    field.ffthelper.pfft! * field.By
    field.ffthelper.pfft! * field.Bz
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
          Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shapes, x[i], y[i])
          @assert all(isfinite, (Exi, Eyi, Ezi, Bxi, Byi, Bzi))
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, q_m);
          @assert all(isfinite, (x[i], y[i], vx[i], vy[i]))
          x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
          y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
          deposit!(J⁰, species.shapes, x[i], y[i], NX_Lx, NY_Ly,
            vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
          x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
          y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
    reduction!(field.Jx⁰, field.Jy⁰, field.Jz⁰, field.Js⁰)
  end

end


function diagnosticsop!(op::F, f::LorenzGaugeField) where F
  op * f.Ax⁰
  op * f.Ay⁰
  op * f.Az⁰
  op * f.ϕ⁰
  op * f.ρ⁰
end

preparefieldsft!(f::LorenzGaugeField) = diagnosticsop!(f.ffthelper.pifft!, f)
restorefieldsft!(f::LorenzGaugeField) = diagnosticsop!(f.ffthelper.pfft!, f)

