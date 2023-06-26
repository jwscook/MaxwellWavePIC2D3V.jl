
struct LorenzGaugeStaggeredField{T, U} <: AbstractLorenzGaugeField
  imex::T
  ρJs⁰::OffsetArray{Float64, 4, Array{Float64, 4}}
  ϕ⁺::Array{ComplexF64, 2}
  ϕ⁰::Array{ComplexF64, 2}
  ϕ⁻::Array{ComplexF64, 2}
  Ax⁺::Array{ComplexF64, 2}
  Ay⁺::Array{ComplexF64, 2}
  Az⁺::Array{ComplexF64, 2}
  Ax⁻::Array{ComplexF64, 2}
  Ay⁻::Array{ComplexF64, 2}
  Az⁻::Array{ComplexF64, 2}
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

function LorenzGaugeStaggeredField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
    imex::AbstractImEx=Explicit(), buffer=0)
  EBxyz = OffsetArray(zeros(6, NX+2buffer, NY+2buffer), 1:6, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  ρJs = OffsetArray(zeros(4, NX+2buffer, NY+2buffer, nthreads()),
    1:4, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer, 1:nthreads());
  return LorenzGaugeStaggeredField(imex, ρJs,
    (zeros(ComplexF64, NX, NY) for _ in 1:22)..., EBxyz,
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end


function warmup!(field::LorenzGaugeStaggeredField, plasma, to)
  ρcallback(a, b, c, d) = (a,)
  Jcallback(a, b, c, d) = (b, c, d)
  @timeit to "Warmup" begin
    dt = timestep(field)
    advect!(plasma, field.gridparams, -dt/2, to)
    field.ρJs⁰ .= 0
    deposit!(field.ρJs⁰, plasma, field.gridparams, dt, to, ρcallback)
    advect!(plasma, field.gridparams, dt/2, to) # back to start, n
    deposit!(field.ρJs⁰, plasma, field.gridparams, dt, to, Jcallback)
    reduction!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰)
  end
end


function loop!(plasma, field::LorenzGaugeStaggeredField, to, t)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)
  # Assume ρ and J are up to date at the current time (n+0)
  # At this point Ai⁰ stores the (n+0)th timestep value and Ai⁻ the (n-1)th
  #               ϕ⁰  stores the (n-1/2)th timestep value and ϕ⁻ the (n-3/2)th
  @timeit to "Field Forward FT" begin
    #@show mean(field.ρ⁰), std(field.ρ⁰)
    #@show mean(field.Jx⁰), std(field.Jx⁰)
    #@show mean(field.Jy⁰), std(field.Jy⁰)
    #@show mean(field.Jz⁰), std(field.Jz⁰)
    field.ffthelper.pfft! * field.ρ⁰;
    field.ffthelper.pfft! * field.Jx⁰;
    field.ffthelper.pfft! * field.Jy⁰;
    field.ffthelper.pfft! * field.Jz⁰;
    #smoothinfourierspace!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ffthelper)
# not necessary given E, B calculated from derivatives of ϕ, Ai
#    field.ρ⁰[1, 1] = field.Jx⁰[1, 1] = field.Jy⁰[1, 1] = field.Jz⁰[1, 1] = 0
  end

  @timeit to "Field Solve" begin
    lorenzgauge!(field.imex, field.ϕ⁺,  field.ϕ⁰,  field.ϕ⁻,  field.ρ⁰,  field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ax⁺, field.Ax⁰, field.Ax⁻, field.Jx⁰, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ay⁺, field.Ay⁰, field.Ay⁻, field.Jy⁰, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Az⁺, field.Az⁰, field.Az⁻, field.Jz⁰, field.ffthelper.k², dt^2)
#    @show mean(field.ϕ⁺), std(field.ϕ⁺)
#    @show mean(field.Ax⁺), std(field.Ax⁺)
#    @show mean(field.Ay⁺), std(field.Ay⁺)
#    @show mean(field.Az⁺), std(field.Az⁺)
  end

  # at this point (ϕ, Ai) stores the (n+1)th timestep value and (ϕ⁻, Ai⁻) the nth
  # Now calculate the value of E and B at n+1/2
  # Eʰ = -∇ϕ⁺ - (A⁺ - A⁰)/dt
  # Bʰ = ∇x(A⁺ + A⁰)/2
  #
  #  E.....E.....E
  #  B.....B.....B
  #  ϕ.....ϕ.....ϕ
  #  -..A..0..A..+..A
  #  ρ.....ρ.....ρ
  #  -..J..0..J..+..J
  #  x.....x.....x
  #  -..v..0..v..+..v
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
#  @show mean(field.Bx), std(field.Bx)
#  @show mean(field.By), std(field.By)
#  @show mean(field.Bz), std(field.Bz)
#  @show mean(field.Ex), std(field.Ex)
#  @show mean(field.Ey), std(field.Ey)
#  @show mean(field.Ez), std(field.Ez)
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
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, q_m);
          x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
          y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
          # deposit ρ at (n+1/2)th timestep
          deposit!(ρJ⁰, species.shape, x[i], y[i], NX_Lx, NY_Ly, qw_ΔV)
          x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
          y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
          # deposit J at the (n+1)th point
          deposit!(ρJ⁰, species.shape, x[i], y[i], NX_Lx, NY_Ly,
            vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
    reduction!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰)
  end

  @timeit to "Copy over buffers" begin
    field.ϕ⁻ .= field.ϕ⁰
    field.ϕ⁰ .= field.ϕ⁺
    field.Ax⁻ .= field.Ax⁰
    field.Ax⁰ .= field.Ax⁺
    field.Ay⁻ .= field.Ay⁰
    field.Ay⁰ .= field.Ay⁺
    field.Az⁻ .= field.Az⁰
    field.Az⁰ .= field.Az⁺
  end
end

@inline function (f::AbstractLorenzGaugeField)(s::AbstractShape, xi, yi)
  NX, NY = f.gridparams.NX, f.gridparams.NY
  NX_Lx, NY_Ly = f.gridparams.NX_Lx, f.gridparams.NY_Ly
  Ex = Ey = Ez = Bx = By = Bz = zero(real(eltype(f.EBxyz)))
  for (j, wy) in depositindicesfractions(s, yi, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, xi, NX, NX_Lx)
      wxy = wx * wy
      @muladd Ex = Ex + f.EBxyz[1,i,j] * wxy
      @muladd Ey = Ey + f.EBxyz[2,i,j] * wxy
      @muladd Ez = Ez + f.EBxyz[3,i,j] * wxy
      @muladd Bx = Bx + f.EBxyz[4,i,j] * wxy
      @muladd By = By + f.EBxyz[5,i,j] * wxy
      @muladd Bz = Bz + f.EBxyz[6,i,j] * wxy
    end
  end
  return (Ex, Ey, Ez, Bx, By, Bz)
end

function update!(f::AbstractLorenzGaugeField)
  f.EBxyz .= 0.0
  applyperiodicity!((@view f.EBxyz[1, :, :]), f.Ex)
  applyperiodicity!((@view f.EBxyz[2, :, :]), f.Ey)
  applyperiodicity!((@view f.EBxyz[3, :, :]), f.Ez)
  applyperiodicity!((@view f.EBxyz[4, :, :]), f.Bx)
  applyperiodicity!((@view f.EBxyz[5, :, :]), f.By)
  applyperiodicity!((@view f.EBxyz[6, :, :]), f.Bz)
  @views for k in axes(f.EBxyz, 3), j in axes(f.EBxyz, 2), i in 1:3
    f.EBxyz[i+3, j, k] += f.B0[i]
  end
end

