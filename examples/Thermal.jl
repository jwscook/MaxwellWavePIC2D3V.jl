using ProgressMeter, TimerOutputs, Plots, FFTW, Random, StaticNumbers
using ThreadPinning, JLD2, ThreadsX

@static if Base.Sys.islinux()
  pinthreads(:cores)
end

include("../src/MaxwellWavePIC2D3V.jl")
import .MaxwellWavePIC2D3V

FFTW.set_num_threads(Threads.nthreads())

Random.seed!(0)
pow(a, b) = a^b
function pic()

  SPEED_OF_LIGHT = 299792458.0
  ELEMENTARY_MASS = 9.1093837e-31;
  ELEMENTARY_CHARGE = 1.60217663e-19;
  MU_0 = 4.0e-7 * π;
  EPSILON_0 = 1.0 / MU_0 / SPEED_OF_LIGHT / SPEED_OF_LIGHT;

  n = 1e19
  B = 2.1
  Mi = 64 #2 * 1836
  Me = 1

  TeeV = 1e4
  Va = B / sqrt(MU_0 * Mi * ELEMENTARY_MASS * n)
  Wp = ELEMENTARY_CHARGE * sqrt(n / Me / ELEMENTARY_MASS / EPSILON_0)
  vthe = sqrt(TeeV * ELEMENTARY_CHARGE * 2 / Me / ELEMENTARY_MASS)
  lD0 = vthe / Wp
  Ωi0 = ELEMENTARY_CHARGE * B / Mi / ELEMENTARY_MASS
  kresolution = 5
  L = Va / Ωi0 * 2π * kresolution
  @show REQUIRED_GRID_CELLS = L / lD0

  @show m_lengthScale = L
  m_timeScale = m_lengthScale / SPEED_OF_LIGHT;
  m_electricPotentialScale = ELEMENTARY_MASS *
                             pow(m_lengthScale / m_timeScale, 2) /
                             ELEMENTARY_CHARGE;
  m_chargeDensityScale =
      m_electricPotentialScale * EPSILON_0 / pow(m_lengthScale, 2);
  m_numberDensityScale = m_chargeDensityScale / ELEMENTARY_CHARGE;
  m_magneticPotentialScale =
      m_timeScale * m_electricPotentialScale / m_lengthScale;
  m_currentDensityScale = m_chargeDensityScale * m_lengthScale / m_timeScale;

  to = TimerOutput()

  NQ = 4
  NX = 2^9 * NQ #10
  NY = 2^9 ÷ NQ #10

  L0 = L / m_lengthScale
  B0 = B / (m_magneticPotentialScale / m_lengthScale)
  n0 = n / m_numberDensityScale
  Πe = sqrt(ELEMENTARY_CHARGE^2 * n / EPSILON_0 / Me / ELEMENTARY_MASS) * m_timeScale
  vth = sqrt(TeeV * ELEMENTARY_CHARGE * 2 / Me / ELEMENTARY_MASS) / m_lengthScale * m_timeScale
  Ωi = B0 / Mi
  #Ωi, Ωi * m_timeScale
  ld = vth / Πe
  #lD0 / m_lengthScale, ld
  Va = Va / SPEED_OF_LIGHT

  #@timeit to "Initialisation" begin
    Lx = L0
    Ly = Lx * NY / NX
    dt = Lx / NX / 2
    P = NX * NY * 8
    NT = 2^12 # 2^14
    Δx = Lx / NX
    Δx = Lx / NX
    Δy = Ly / NY
    dl = min(Lx / NX, Ly / NY)
    #n0 = 3.5e6 #4 * pi^2
    #debyeoverresolution = 1
    #vth = 0.01 #debyeoverresolution * dl * sqrt(n0)
    #B0 = sqrt(n0) / 4;
    #λ = vth / sqrt(n0)
    #rL = vth / B0 = 4λ

    #dt = dl/6vth
    #field = MaxwellWavePIC2D3V.ElectrostaticField(NX, NY, Lx, Ly, dt=dt, B0x=B0)
    #diagnostics = MaxwellWavePIC2D3V.ElectrostaticDiagnostics(NX, NY, NT, ntskip, 2)
    ntskip = 16 #4#prevpow(2, round(Int, 10 / 6vth)) ÷ 4
    ngskip = 4
    @show NT ÷ ntskip
    #dt = 2dl #/6vth
    #dt = dl / vth
    field = MaxwellWavePIC2D3V.LorenzGaugeField(NX, NY, Lx, Ly, dt=dt, B0y=B0,
      imex=MaxwellWavePIC2D3V.ImEx(1), buffers=10)
    #field = MaxwellWavePIC2D3V.EJField(NX, NY, Lx, Ly, dt=dt, B0y=B0,
    #  imex=MaxwellWavePIC2D3V.ImEx(1), buffers=10)
    #field = MaxwellWavePIC2D3V.LorenzGaugeSemiImplicitField(NX, NY, Lx, Ly, dt=dt, B0y=B0,
    #  fieldimex=MaxwellWavePIC2D3V.ImEx(1.0), sourceimex=MaxwellWavePIC2D3V.ImEx(0.05),
    #  buffers=10, rtol=100eps(), maxiters=1000)
    diagnostics = MaxwellWavePIC2D3V.LorenzGaugeDiagnostics(NX, NY, NT, ntskip, ngskip; makegifs=false)
    shape = MaxwellWavePIC2D3V.BSplineWeighting{@stat 5}()
    #shape = MaxwellWavePIC2D3V.NGPWeighting();#
    #shape = MaxwellWavePIC2D3V.AreaWeighting();#
    electrons = MaxwellWavePIC2D3V.Species(P, vth, n0, shape;
      Lx=Lx, Ly=Ly, charge=-1, mass=Me)
    ions = MaxwellWavePIC2D3V.Species(P, vth / sqrt(Mi / Me), n0, shape;
      Lx=Lx, Ly=Ly, charge=1, mass=Mi)
    sort!(electrons, Lx / NX, Ly / NY)
    sort!(ions, Lx / NX, Ly / NY)
    plasma = [electrons, ions]
    #@show NX, NY, P, NT, NT÷ntskip, ntskip, dl, n0, vth, B0, dt
    #@show vth * (NT * dt)
    #@show (NT * dt) / (2pi/B0), (2pi/B0) / (dt * ntskip)
    #@show (NT * dt) / (2pi/sqrt(n0)),  (2pi/sqrt(n0)) / (dt * ntskip)
#  end

  MaxwellWavePIC2D3V.printresolutions(plasma, field, dt, NT, to)
  MaxwellWavePIC2D3V.warmup!(field, plasma, to)

  progress = Progress(NT; showspeed=true)
  for t in 0:NT-1;
    MaxwellWavePIC2D3V.loop!(plasma, field, to, t)
    MaxwellWavePIC2D3V.diagnose!(diagnostics, field, plasma, t, to)
    ProgressMeter.next!(progress;
      showvalues=[(:t,t), (:energy, diagnostics.totalenergydensity),
                  (:momentum, diagnostics.totalmomentumdensitychange)])
    #t % 2^12 == 0 && ThreadsX.map(s->sort!(s, Lx / NX, Ly / NY), plasma)
  end

  show(to)

  return diagnostics, field, plasma, n0, Va, Ωi, NT, B0
end
using StatProfilerHTML
diagnostics, field, plasma, n0, vcharacteristic, omegacharacteristic, NT, B0 = pic()


MaxwellWavePIC2D3V.plotfields(diagnostics, field, n0, vcharacteristic, omegacharacteristic, NT;
                              cutoff=10 * omegacharacteristic)

const filecontents = [i for i in readlines(open(@__FILE__))]

@save "$(hash(filecontents))" filecontents diagnostics field plasma n0 vcharacteristic omegacharacteristic NT B0


