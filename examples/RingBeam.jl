using ProgressMeter, TimerOutputs, Plots, FFTW, Random, StaticNumbers
using ThreadPinning, JLD2

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

  n = 1e20
  B = 2.1
  Mi = 64 #2 * 1836
  Me = 1

  TeeV = 1e4
  Va = B / sqrt(MU_0 * Mi * ELEMENTARY_MASS * n)
  Wp = ELEMENTARY_CHARGE * sqrt(n / Me / ELEMENTARY_MASS / EPSILON_0)
  vthe = sqrt(TeeV * ELEMENTARY_CHARGE * 2 / Me / ELEMENTARY_MASS)
  lD0 = vthe / Wp
  Ωi0 = ELEMENTARY_CHARGE * B / Mi / ELEMENTARY_MASS
  kresolution = 1
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
  Va = Va / SPEED_OF_LIGHT

  Ωi = B0 / Mi
  ld = vth / Πe

  Lx = L0
  Ly = Lx * NY / NX
  dt = Lx / NX / 8
  P = NX * NY * 8
  NT = 2^14
  Δx = Lx / NX
  Δx = Lx / NX
  Δy = Ly / NY
  dl = min(Lx / NX, Ly / NY)

  ntskip = 16
  ngskip = 4
  @show NT ÷ ntskip
  field = MaxwellWavePIC2D3V.LorenzGaugeField(NX, NY, Lx, Ly, dt=dt, B0y=B0,
    imex=MaxwellWavePIC2D3V.ImEx(1), buffer=10)
  #field = MaxwellWavePIC2D3V.LorenzGaugeSemiImplicitField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
  #  fieldimex=MaxwellWavePIC2D3V.ImEx(1.0), sourceimex=MaxwellWavePIC2D3V.ImEx(0.05),
  #  buffer=10, rtol=sqrt(eps()), maxiters=1000)
  diagnostics = MaxwellWavePIC2D3V.LorenzGaugeDiagnostics(NX, NY, NT, ntskip, ngskip; makegifs=false)
  shape = MaxwellWavePIC2D3V.BSplineWeighting{@stat 5}()
  electrons = MaxwellWavePIC2D3V.Species(P, vth, n0, shape;
    Lx=Lx, Ly=Ly, charge=-1, mass=Me)
  ions = MaxwellWavePIC2D3V.Species(P, vth / sqrt(Mi / Me), n0, shape;
    Lx=Lx, Ly=Ly, charge=1, mass=Mi)
  sort!(electrons, Lx / NX, Ly / NY)
  sort!(ions, Lx / NX, Ly / NY)
  plasma = [electrons, ions]

  MaxwellWavePIC2D3V.printresolutions(plasma, field, dt, NT, to)
  for t in 0:NT-1;
    MaxwellWavePIC2D3V.loop!(plasma, field, to, t)
    MaxwellWavePIC2D3V.diagnose!(diagnostics, field, plasma, t, to)
  end

  show(to)

  return diagnostics, field, plasma, n0, Va, Ωi, NT, B0
end
using StatProfilerHTML
diagnostics, field, plasma, n0, vcharacteristic, omegacharacteristic, NT, B0 = pic()


MaxwellWavePIC2D3V.plotfields(diagnostics, field, n0, vcharacteristic, omegacharacteristic, NT;
                              cutoff=20 * omegacharacteristic)

const filecontents = [i for i in readlines(open(@__FILE__))]

@save "$(hash(filecontents)" filecontents diagnostics field plasma n0 vcharacteristic omegacharacteristic NT B0

