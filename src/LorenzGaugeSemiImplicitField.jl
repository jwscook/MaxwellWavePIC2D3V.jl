
function warmup!(field::LorenzGaugeSemiImplicitField, plasma, to)
  @timeit to "Warmup" begin
    dt = timestep(field)
    warmup!(field.ρ⁻, field.Jx⁻, field.Jy⁻, field.Jz⁻, field.Js⁻, plasma, field.gridparams, -dt, to)
    warmup!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.Js⁰, plasma, field.gridparams, dt, to)
    warmup!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.Js⁺, plasma, field.gridparams, dt, to)
    advect!(plasma, field.gridparams, -dt, to) # advect back to start
  end
end


struct LorenzGaugeSemiImplicitField{T, U, V} <: AbstractLorenzGaugeField
  fieldimex::T
  sourceimex::U
  Js⁻::OffsetArray{Float64, 4, Array{Float64, 4}}
  Js⁰::OffsetArray{Float64, 4, Array{Float64, 4}}
  Js⁺::OffsetArray{Float64, 4, Array{Float64, 4}}
  ρJsᵗ::OffsetArray{Float64, 4, Array{Float64, 4}}
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
  ρ⁻::Array{ComplexF64, 2}
  ρ⁰::Array{ComplexF64, 2}
  ρ⁺::Array{ComplexF64, 2}
  Jx⁻::Array{ComplexF64, 2}
  Jy⁻::Array{ComplexF64, 2}
  Jz⁻::Array{ComplexF64, 2}
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
  ffthelper::V
  boris::ElectromagneticBoris
  dt::Float64
  rtol::Float64
  maxiters::Int
end

function LorenzGaugeSemiImplicitField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
    fieldimex::AbstractImEx=Explicit(), sourceimex::AbstractImEx=Explicit(),
    buffer=0, rtol=sqrt(eps()), maxiters=10)
  EBxyz = OffsetArray(zeros(6, NX+2buffer, NY+2buffer), 1:6, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  Js = OffsetArray(zeros(4, NX+2buffer, NY+2buffer, nthreads()),
    1:3, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer, 1:nthreads());
  return LorenzGaugeSemiImplicitField(fieldimex, sourceimex, Js, deepcopy(Js), deepcopy(Js),
    deepcopy(Js), (zeros(ComplexF64, NX, NY) for _ in 1:30)..., EBxyz,
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt, rtol, maxiters)
end

function loop!(plasma, field::LorenzGaugeSemiImplicitField, to, t, plasmacopy = deepcopy(plasma))
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)

  copyto!.(plasmacopy, plasma)
  firstloop = true
  iters = 0
  while true
    if (iters > 0) && (iters > field.maxiters || isapprox(field.ρJsᵗ, field.Js⁺, rtol=field.rtol, atol=0))
      for species in plasma
        x = positions(species)
        v = velocities(species)
        xʷ = positions(species; work=true)
        vʷ = velocities(species; work=true)
        @tturbo x .= xʷ
        @tturbo v .= vʷ
      end
      break
    end
    iters += 1
    copyto!.(plasma, plasmacopy)
    @tturbo field.Jsᵗ .= field.Js⁺
    @tturbo field.Js⁺ .= 0
    @timeit to "Particle Loop" begin
      @threads for j in axes(field.Js⁺, 4)
        ρJ⁺ = @view field.Js⁺[:, :, :, j]
        for species in plasma
          qw_ΔV = species.charge * species.weight / ΔV
          q_m = species.charge / species.mass
          x = @view positions(species)[1, :]
          y = @view positions(species)[2, :]
          vx = @view velocities(species)[1, :]
          vy = @view velocities(species)[2, :]
          vz = @view velocities(species)[3, :]
          xʷ = @view positions(species; work=true)[1, :]
          yʷ = @view positions(species; work=true)[2, :]
          vxʷ = @view velocities(species; work=true)[1, :]
          vyʷ = @view velocities(species; work=true)[2, :]
          vzʷ = @view velocities(species; work=true)[3, :]
          for i in species.chunks[j]
            Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
            xʷ[i] = unimod(x[i] + vx[i] * dt, Lx)
            yʷ[i] = unimod(y[i] + vy[i] * dt, Ly)
            vxʷ[i], vyʷ[i], vzʷ[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
              Bxi, Byi, Bzi, q_m);
            # now deposit ρ at (n+1)th timestep
            deposit!(ρJ⁺, species.shape, xʷ[i], yʷ[i], NX_Lx, NY_Ly,
              vxʷ[i] * qw_ΔV, vyʷ[i] * qw_ΔV,  vzʷ[i] * qw_ΔV)
          end
        end
      end
    end
    @timeit to "Field Reduction" begin
      reduction!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.Js⁺)
      #field.Js⁺ .= 0 # dont zero it here!
    end
    @timeit to "Field Forward FT" begin
      field.ffthelper.pfft! * field.ρ⁺;
      field.ffthelper.pfft! * field.Jx⁺;
      field.ffthelper.pfft! * field.Jy⁺;
      field.ffthelper.pfft! * field.Jz⁺;
      field.ρ⁺[1, 1] = 0
      field.Jx⁺[1, 1] = 0
      field.Jy⁺[1, 1] = 0
      field.Jz⁺[1, 1] = 0
    end
    @timeit to "Field Solve" begin
      chargeconservation!(field.ρ⁰, field.ρ⁻, field.Jx⁰, field.Jy⁰, field.ffthelper, dt)
      # at this point ϕ stores the nth timestep value and ϕ⁻ the (n-1)th
      lorenzgauge!(field.fieldimex, field.ϕ⁺, field.ϕ⁰,  field.ϕ⁻, field.ρ⁺, field.ρ⁰, field.ρ⁻, field.ffthelper.k², dt^2, field.sourceimex)
      lorenzgauge!(field.fieldimex, field.Ax⁺, field.Ax⁰, field.Ax⁻, field.Jx⁺, field.Jx⁰, field.Jx⁻, field.ffthelper.k², dt^2, field.sourceimex)
      lorenzgauge!(field.fieldimex, field.Ay⁺, field.Ay⁰, field.Ay⁻, field.Jy⁺, field.Jy⁰, field.Jy⁻, field.ffthelper.k², dt^2, field.sourceimex)
      lorenzgauge!(field.fieldimex, field.Az⁺, field.Az⁰, field.Az⁻, field.Jz⁺, field.Jz⁰, field.Jz⁻, field.ffthelper.k², dt^2, field.sourceimex)
    end
    @timeit to "Calculate E, B" begin
      θ = theta(field.fieldimex)
      @. field.Ex = -im * field.ffthelper.kx * (θ/2 * (field.ϕ⁺ + field.ϕ⁻) + (1-θ)*field.ϕ⁰)
      @. field.Ey = -im * field.ffthelper.ky * (θ/2 * (field.ϕ⁺ + field.ϕ⁻) + (1-θ)*field.ϕ⁰)
      @. field.Ex -= (field.Ax⁺ - field.Ax⁻)/2dt
      @. field.Ey -= (field.Ay⁺ - field.Ay⁻)/2dt
      @. field.Ez = -(field.Az⁺ - field.Az⁻)/2dt
      @. field.Bx =  im * field.ffthelper.ky * (θ/2 * (field.Az⁺ + field.Az⁻) + (1-θ)*field.Az⁰)
      @. field.By = -im * field.ffthelper.kx * (θ/2 * (field.Az⁺ + field.Az⁻) + (1-θ)*field.Az⁰)
      @. field.Bz =  im * field.ffthelper.kx * (θ/2 * (field.Ay⁺ + field.Ay⁻) + (1-θ)*field.Ay⁰)
      @. field.Bz -= im * field.ffthelper.ky * (θ/2 * (field.Ax⁺ + field.Ax⁻) + (1-θ)*field.Ax⁰)
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
  end
  @timeit to "Copy over buffers" begin
    field.Js⁻ .= field.Js⁰
    field.Js⁰ .= field.Js⁺
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


