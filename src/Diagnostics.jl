abstract type AbstractDiagnostics end

struct ElectrostaticDiagnostics <: AbstractDiagnostics
  kineticenergy::Vector{Float64}
  fieldenergy::Vector{Float64}
  particlemomentum::Vector{Vector{Float64}}
  characteristicmomentum::Vector{Vector{Float64}}
  Exs::Array{Float64, 3}
  Eys::Array{Float64, 3}
  ϕs::Array{Float64, 3}
  ntskip::Int
  ngskip::Int
  ti::Ref{Int64}
  makegifs::Bool
  totalenergy::Ref{Float64}
  totalmomentum::Vector{Float64}
end

function generatestorage(NX, NY, ND, nscalar, nmomentum, nstorage)
  scalarstorage = (zeros(ND) for _ in 1:nscalar)
  momentumstorage = ([zeros(3) for _ in 1:ND] for _ in 1:nmomentum)
  fieldstorage = (zeros(NX, NY, ND) for _ in 1:nstorage)
  return (scalarstorage, momentumstorage, fieldstorage)
end

function ElectrostaticDiagnostics(NX, NY, NT, ntskip, ngskip=1; makegifs=false)
  @assert NT >= ntskip
  @assert ispow2(ngskip)
  scalarstorage, momentumstorage, fieldstorage = generatestorage(
    NX÷ngskip, NY÷ngskip, NT÷ntskip, 2, 2, 3)
  return ElectrostaticDiagnostics(scalarstorage..., momentumstorage...,
    fieldstorage..., ntskip, ngskip, Ref(0), makegifs, Ref(0.0), zeros(3))
end

struct LorenzGaugeDiagnostics <: AbstractDiagnostics
  kineticenergy::Array{Float64, 1}
  fieldenergy::Array{Float64, 1}
  particlemomentum::Vector{Vector{Float64}}
  fieldmomentum::Vector{Vector{Float64}}
  characteristicmomentum::Vector{Vector{Float64}}
  Exs::Array{Float64, 3}
  Eys::Array{Float64, 3}
  Ezs::Array{Float64, 3}
  Bxs::Array{Float64, 3}
  Bys::Array{Float64, 3}
  Bzs::Array{Float64, 3}
  Axs::Array{Float64, 3}
  Ays::Array{Float64, 3}
  Azs::Array{Float64, 3}
  ϕs::Array{Float64, 3}
  ρs::Array{Float64, 3}
  Jxs::Array{Float64, 3}
  Jys::Array{Float64, 3}
  Jzs::Array{Float64, 3}
  ntskip::Int
  ngskip::Int
  ti::Ref{Int64}
  makegifs::Bool
  totalenergy::Ref{Float64}
  totalmomentum::Vector{Float64}
end

function LorenzGaugeDiagnostics(NX, NY, NT::Int, ntskip::Int, ngskip=1;
                                makegifs=false)
  @assert NT >= ntskip
  @assert ispow2(ngskip)
  scalarstorage, momentumstorage, fieldstorage = generatestorage(
    NX÷ngskip, NY÷ngskip, NT÷ntskip, 2, 3, 14)
  return LorenzGaugeDiagnostics(scalarstorage..., momentumstorage...,
    fieldstorage..., ntskip, ngskip, Ref(0), makegifs, Ref(0.0), zeros(3))
end



function diagnose!(d::AbstractDiagnostics, plasma, to)
  @timeit to "Plasma" begin
    ti = d.ti[]
    d.kineticenergy[ti] = sum(kineticenergy(s) for s in plasma)
    d.particlemomentum[ti] .= sum(momentum(s) for s in plasma)
    d.characteristicmomentum[ti] .= sum(characteristicmomentum(s) for s in plasma)
  end
end

function diagnose!(d::ElectrostaticDiagnostics, f::ElectrostaticField, plasma,
                   t, to)
  @timeit to "Diagnostics" begin
    t % d.ntskip == 0 && (d.ti[] += 1)
    if t % d.ntskip == 0
      diagnose!(d, plasma, to)
    end
    @timeit to "Fields" begin
      ti = d.ti[]
      if t % d.ntskip == 0
        d.fieldenergy[ti] = mean(abs2, f.Exy) / 2
      end
      a = 1:d.ngskip:size(f.Ex, 1)
      b = 1:d.ngskip:size(f.Ex, 2)
      @views d.Exs[:, :, ti] .+= real.(f.Ex[a, b]) ./ d.ntskip
      @views d.Eys[:, :, ti] .+= real.(f.Ey[a, b]) ./ d.ntskip
      f.ffthelper.pifft! * f.ϕ
      @views d.ϕs[:, :, ti] .+= real.(f.ϕ)[a, b] ./ d.ntskip
      f.ffthelper.pfft! * f.ϕ
    end
  end
end

function preparefieldsft!(f::AbstractLorenzGaugeField)
  f.ffthelper.pifft! * f.Ax⁰
  f.ffthelper.pifft! * f.Ay⁰
  f.ffthelper.pifft! * f.Az⁰
  f.ffthelper.pifft! * f.ϕ⁰
  #f.ffthelper.pifft! * f.ρ⁰
  #f.ffthelper.pifft! * f.Jx⁰
  #f.ffthelper.pifft! * f.Jy⁰
  #f.ffthelper.pifft! * f.Jz⁰
end
function restorefieldsft!(f::AbstractLorenzGaugeField)
  f.ffthelper.pfft! * f.Ax⁰
  f.ffthelper.pfft! * f.Ay⁰
  f.ffthelper.pfft! * f.Az⁰
  f.ffthelper.pfft! * f.ϕ⁰
  #f.ffthelper.pifft! * f.ρ⁰
  #f.ffthelper.pifft! * f.Jx⁰
  #f.ffthelper.pifft! * f.Jy⁰
  #f.ffthelper.pifft! * f.Jz⁰
end




function diagnose!(d::LorenzGaugeDiagnostics, f::AbstractLorenzGaugeField, plasma,
    t, to)
  @timeit to "Diagnostics" begin
    t % d.ntskip == 0 && (d.ti[] += 1)
    if t % d.ntskip == 0
      diagnose!(d, plasma, to)
    end
    @timeit to "Fields" begin
      ti = d.ti[]
      if t % d.ntskip == 0
        @timeit to "Energy" begin
          d.fieldenergy[ti] = mean(abs2, f.EBxyz) / 2
        end
        @timeit to "Momentum" begin
          px, py, pz = 0.0, 0.0, 0.0
          @assert isapprox(abs(mean(f.Bx)), 0.0, atol=sqrt(eps()) * mean(f.B0))
          @assert isapprox(abs(mean(f.By)), 0.0, atol=sqrt(eps()) * mean(f.B0))
          @assert isapprox(abs(mean(f.Bz)), 0.0, atol=sqrt(eps()) * mean(f.B0))
          for i in eachindex(f.Ex)
            px += real(f.Ey[i]) * real(f.Bz[i] .+ f.B0[3])
                - real(f.Ez[i]) * real(f.By[i] .+ f.B0[2])
            py += real(f.Ez[i]) * real(f.Bx[i] .+ f.B0[1])
                - real(f.Ex[i]) * real(f.Bz[i] .+ f.B0[3])
            pz += real(f.Ex[i]) * real(f.By[i] .+ f.B0[2])
                - real(f.Ey[i]) * real(f.Bx[i] .+ f.B0[1])
          end
          d.fieldmomentum[ti] .= (px, py, pz) ./ length(f.Ex)
        end
        totenergy = (d.fieldenergy[ti] + d.kineticenergy[ti]) / (d.fieldenergy[1] + d.kineticenergy[1])
        totmomentum = (d.fieldmomentum[ti] + d.particlemomentum[ti]) ./ mean(d.characteristicmomentum[1])
        d.totalenergy[] = totenergy
        d.totalmomentum .= totmomentum
      end
      @timeit to "Prepare fields (i)fft!" begin
        preparefieldsft!(f)
      end
      @timeit to "Field averaging" begin
        function average!(lhs, rhs)
          a = 1:d.ngskip:size(rhs, 1)
          b = 1:d.ngskip:size(rhs, 2)
          factor = 1 / (d.ntskip * d.ngskip^2)
          for (jl, jr) in enumerate(b), (il, ir) in enumerate(a)
            for jj in 0:d.ngskip-1, ii in 0:d.ngskip-1
              lhs[il, jl, ti] += real(rhs[ir+ii, jr+jj]) * factor
            end
          end
        end
        t0 = Threads.@spawn average!(d.Exs, f.Ex)
        t1 = Threads.@spawn average!(d.Eys, f.Ey)
        t2 = Threads.@spawn average!(d.Ezs, f.Ez)
        t3 = Threads.@spawn average!(d.Bxs, f.Bx)
        t4 = Threads.@spawn average!(d.Bys, f.By)
        t5 = Threads.@spawn average!(d.Bzs, f.Bz)
        t6 = Threads.@spawn average!(d.Axs, f.Ax⁰)
        t7 = Threads.@spawn average!(d.Ays, f.Ay⁰)
        t8 = Threads.@spawn average!(d.Azs, f.Az⁰)
        t9 = Threads.@spawn average!(d.ϕs, f.ϕ⁰)
        ta = Threads.@spawn average!(d.ρs, f.ρ⁰)
        tb = Threads.@spawn average!(d.Jxs, f.Jx⁰)
        tc = Threads.@spawn average!(d.Jys, f.Jy⁰)
        td = Threads.@spawn average!(d.Jzs, f.Jz⁰)
        wait.((t0, t1, t2, t3, t4, t5, t6,t7, t8, t9, ta, tb, tc, td))
      end
      @timeit to "Restore fields (i)fft!" begin
        restorefieldsft!(f)
      end
    end
  end
end

function diagnosticfields(d::ElectrostaticDiagnostics)
  return ((d.Exs, "Ex"), (d.Eys, "Ey"), (d.ϕs, "ϕ"))
end

function diagnosticfields(d::LorenzGaugeDiagnostics)
  return ((d.Exs, "Ex"), (d.Eys, "Ey"), (d.Ezs, "Ez"),
          (d.Bxs, "Bx"), (d.Bys, "By"), (d.Bzs, "Bz"),
          (d.Axs, "Ax"), (d.Ays, "Ay"), (d.Azs, "Az"),
          (d.Jxs, "Jx"), (d.Jys, "Jy"), (d.Jzs, "Jz"),
          (d.ϕs, "ϕ"), (d.ρs, "ρ"))
end

function plotfields(d::AbstractDiagnostics, field, n0, vc, w0, NT; cutoff=Inf)
  B0 = norm(field.B0)
  dt = timestep(field)
  g = field.gridparams
  NXd = g.NX÷d.ngskip
  NYd = g.NY÷d.ngskip
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  xs = collect(1:NXd) ./ NXd ./ (vc / w0) * Lx
  ys = collect(1:NYd) ./ NYd ./ (vc / w0) * Ly
  ndiags = d.ti[]
  ts = collect(1:ndiags) .* ((NT * dt / ndiags) / (2pi/w0))

  filter = sin.((collect(1:ndiags) .- 0.5) ./ ndiags .* pi)'
  ws = 2π / (NT * dt) .* (1:ndiags) ./ w0;

  kxs = 2π/Lx .* collect(0:NXd-1) .* (vc / w0);
  kys = 2π/Ly .* collect(0:NYd-1) .* (vc / w0);

  energy0 = d.fieldenergy[1] + d.kineticenergy[1]

  plot(ts, d.fieldenergy ./ energy0, label="Fields")
  plot!(ts, d.kineticenergy ./ energy0, label="Particles")
  plot!(ts, (d.fieldenergy + d.kineticenergy) ./ energy0, label="Total")
  savefig("Energies.png")

  fieldmom = cat(d.fieldmomentum..., dims=2)'
  particlemom = cat(d.particlemomentum..., dims=2)'
  characteristicmom = cat(d.characteristicmomentum..., dims=2)'
  p0 = characteristicmom[1]
  plot(ts, fieldmom ./ p0, label="Fields")
  plot!(ts, particlemom ./ p0, label="Particles")
  plot!(ts, (fieldmom .+ particlemom) ./ p0, label="Total")
  savefig("Momenta.png")


  wind = findlast(ws .< cutoff)
  isnothing(wind) && (wind = length(ws)÷2)
  wind = min(wind, length(ws)÷2)
  @show (ws[1:2], ws[end], cutoff, wind, ws[wind])

  kxind = min(length(kxs)÷2-1, 128)
  kyind = min(length(kys)÷2-1, 128)
  @views for (F, FS) in diagnosticfields(d)
    all(iszero, F) && (println("$FS is empty"); continue)
    if d.makegifs
      maxabsF = maximum(abs, F)
      maxabsF = iszero(maxabsF) ? 1.0 : maxabsF
      nsx = ceil(Int, size(F,1) / 128)
      nsy = ceil(Int, size(F,2) / 128)
      anim = @animate for i in axes(F, 3)
        heatmap(xs[1:nsx:end], ys[1:nsy:end], F[1:nsx:end, 1:nsy:end, i] ./ maxabsF)
        xlabel!(L"Position x $[V_{A} / \Omega]$");
        ylabel!(L"Position y $[V_{A} / \Omega]$")
      end
      gif(anim, "PIC2D3V_$(FS)_XY.gif", fps=10)
    end

#    heatmap(xs, ys, F[:, :, 1])
#    xlabel!(L"Position x $[v_{A} / \Omega]$");
#    ylabel!(L"Position y $[v_{A} / \Omega]$")
#    savefig("PIC2D3V_$(FS)_XY_ic.png")
#
#    heatmap(xs, ys, F[:, :, end])
#    xlabel!(L"Position x $[v_{A} / \Omega]$");
#    ylabel!(L"Position y $[v_{A} / \Omega]$")
#    savefig("PIC2D3V_$(FS)_XY_final.png")

    try
      Z = F[:, :, end]'
      heatmap(xs, ys, Z)
      xlabel!(L"Position x $[V_{A} / \Omega_{i}]$");
      ylabel!(L"Position y $[V_{A} / \Omega_{i}]$");
      savefig("PIC2D3V_$(FS)_XY_t_end.png")
    catch e
      @info "XY_t_end failed for $FS"
      @info e
    end

    try
      Z = log10.(abs.(fft(F)[2:kxind, 1, 1:wind]))'
      heatmap(kxs[2:kxind], ws[1:wind], Z)
      xlabel!(L"Wavenumber x $[\Omega_{i} / V_{A}]$");
      ylabel!(L"Frequency $[\Omega_{i}]$")
      savefig("PIC2D3V_$(FS)_WKx.png")
    catch e
      @info "WKx failed for $FS"
      @info e
    end
    try
      Z = log10.(abs.(fft(F)[1, 2:kyind, 1:wind]))'
      heatmap(kys[2:kyind], ws[1:wind], Z)
      xlabel!(L"Wavenumber y $[\Omega_{i} / V_{A}]$");
      ylabel!(L"Frequency $[\Omega_{i}]$")
      savefig("PIC2D3V_$(FS)_WKy.png")
    catch e
      @info "WKy failed for $FS"
      @info e
    end
  end

end


