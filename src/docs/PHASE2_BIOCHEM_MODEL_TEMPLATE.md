# Phase2 Biochem Model Template (COMSOL Reference)

This document captures the Phase2 biochemistry model template and COMSOL-aligned configuration reference exactly as provided.

## MODEL STRUCTURE

```text
=== MODEL STRUCTURE ===

Component: comp1
  Physics interfaces: 3
    - spf (LaminarFlow)
      * fp1 (FluidProperties)
        -> MixingLengthLimit: Automatic
        -> l_mix_lim: 1
        -> LengthScaleSpecification: Automatic
        -> lref: 1
        -> rho_mat: userdef
        -> rho: rho_b
        -> m_pow: 1
        -> n_pow: 1
        -> sr_lowlimit: 0.01[1/s]
        -> mu0: 0.56*(mu2(FI)+mu1(Mat))
        -> mu_inf: mu_b*(mu2(FI)+mu1(Mat))
        -> n_car: 0.3568
        -> lam_car: 3.313
        -> nonNewtonianModels: Carreau
        -> m_p: 0
        -> tau_y: 0
        -> mu_p: 0
        -> mu_mat: userdef
        -> mu: mu_b*mu(Mat)
        -> streamlinens: 0
        -> crosswindns: 0
        -> Constitutiverelation: InelasticNonNewtonian
        -> sr_ref: 1[1/s]
        -> mu_init_app: spf.mu0
        -> kf: 1[1/s]
        -> kb: 1[1/s]
        -> m_y0: 1e-3[Pa*s]
        -> m_yt: 0[Pa*s]
        -> tau_y0: 0[N/m^2]
        -> tau_yt: 0[N/m^2]
        -> nb: 1
        -> nf: 1
        -> m_pow_mat: from_mat
        -> n_pow_mat: from_mat
        -> mu0_mat: userdef
        -> mu_inf_mat: userdef
        -> lam_car_mat: userdef
        -> n_car_mat: userdef
        -> a_car_mat: from_mat
        -> a_car: 2
        -> tau_y_mat: from_mat
        -> mu_p_mat: from_mat
        -> tau_half_mat: from_mat
        -> tau_half: 1[N/m^2]
        -> a_e_mat: from_mat
        -> a_e: 2
        -> mu_DK_mat: from_mat
        -> mu_DK: 1e-3[Pa*s]
        -> lam_DK_mat: from_mat
        -> lam_DK: 0[s]
        -> tau_tr_mat: from_mat
        -> tau_tr: 1[N/m^2]
        -> thermalFunction: none
        -> Twlf: 293.15[K]
        -> C1wlf: 8.86
        -> C2wlf: 51.6[K]
        -> T0: 293.15[K]
        -> Q: 0
        -> be: 0
        -> alphaT: 1
        -> shift_temperature_src: userdef
        -> shift_temperature: spf.Tref
        -> mu0cy_mat: from_mat
        -> mu0cy: 1e-3[Pa*s]
        -> mu_infcy_mat: from_mat
        -> mu_infcy: 0[Pa*s]
        -> lam_cy_mat: from_mat
        -> lam_cy: 0[s]
        -> n_cy_mat: from_mat
        -> n_cy: 1
        -> mu0cw_mat: from_mat
        -> mu0cw: 1e-3[Pa*s]
        -> lam_cw_mat: from_mat
        -> lam_cw: 0[s]
        -> n_cw_mat: from_mat
        -> n_cw: 1
        -> mu_infs_mat: from_mat
        -> mu_infs: 0[Pa*s]
        -> m_pows_mat: from_mat
        -> m_pows: 1e-3[Pa*s]
        -> n_pows_mat: from_mat
        -> n_pows: 1
        -> mu0e_mat: from_mat
        -> mu0e: 1e-3[Pa*s]
        -> m_powhb_mat: from_mat
        -> m_powhb: 1e-3[Pa*s]
        -> n_powhb_mat: from_mat
        -> n_powhb: 1
        -> tau_yhb_mat: from_mat
        -> tau_yhb: 0[N/m^2]
        -> mu_pc_mat: from_mat
        -> mu_pc: 1e-3[Pa*s]
        -> tau_yc_mat: from_mat
        -> tau_yc: 0[N/m^2]
        -> mu0c_mat: from_mat
        -> mu0c: 1e-3[Pa*s]
        -> mu_infc_mat: from_mat
        -> mu_infc: 0[Pa*s]
        -> n_c_mat: from_mat
        -> n_c: 1
        -> tau_yDK_mat: from_mat
        -> tau_yDK: 0[N/m^2]
        -> m_powrs_mat: from_mat
        -> m_powrs: 1e-3[Pa*s]
        -> n_powrs_mat: from_mat
        -> n_powrs: 1
        -> tau_yrs_mat: from_mat
        -> tau_yrs: 0[N/m^2]
      * init1 (init)
        -> u_init: [0, 0, 0]
        -> p_init: 0
        -> nutilde_init: spf.nutildeinit
        -> G_init: spf.G0
        -> k_init: spf.kinit
        -> ep_init: spf.epinit
        -> om_init: spf.omInit
        -> yPlus_init: spf.yPlusinit
        -> uPlus_init: spf.uPlusinit
        -> zeta_init: 2/3
        -> alpha_init: 1
        -> gamma_init: 1
        -> R_init: (2/3)*spf.kinit
      * wallbc1 (WallBC)
        -> BoundaryCondition: NoSlip
        -> utr: [0, 0, 0]
        -> uvw: 0
        -> uleak: [0, 0, 0]
        -> ElectroosmoticOption: userdef
        -> mueo: 7e-8[m^2/(V*s)]
        -> zeta: (-0.1)[V]
        -> epsilonr: 80
        -> UseViscousSlip: 0
        -> SlipLengthOption: userdef
        -> Ls: 1e-7[m]
        -> alphav: 0.9
        -> lambda: 1e-6[m]
        -> UseThermalCreep: 0
        -> TranslationalVelocityOption: AutomaticFromFrame
        -> SlidingWall: 0
        -> sigmat: 0.75
        -> NavierSlip: hmin
        -> beta: 1[mm]
        -> beta_factor: 0.5
        -> E_src: userdef
        -> E: [0, 0, 0]
        -> T_src: userdef
        -> T: 293.15[K]
        -> ApplyWallRoughness: 0
        -> RoughnessModel: SandRoughness
        -> kseq: 3.2[um]
        -> ks: 3.2[um]
        -> Cs: 0.26
        -> tau_ref: 1[N/m^2]
        -> m: 1
        -> IncludeUtrInNavierSlipForce: 0
        -> tau_y: 1[N/m^2]
        -> k1: 1[m/s]
        -> k2: 1[1/Pa]
      * inl1 (InletBoundary)
        -> BoundaryCondition: FullyDevelopedFlow
        -> ComponentWise: NormalInflowVelocity
        -> U0in: inlet(x)
        -> u0: [0, 0, 0]
        -> p0: 800*stept(t)
        -> SuppressBackflow: 0
        -> FlowDirection: NormalFlow
        -> d_u: [1, 0, 0]
        -> LaminarInflowOption: Uav
        -> Uav: 0
        -> V0: 0
        -> p0_entr: 0
        -> Lentr: 1
        -> Dzentr: 1
        -> ConstrainEndPointsToZero: 0
        -> FullyDevelopedFlowOption: Uav
        -> Uavfdf: U_inlet
        -> V0fdf: 0
        -> p0avfdf: 0
        -> Dzfdf: 1
        -> MassFlowType: MassFlowRate
        -> mfr: 1e-5[kg/s]
        -> StandardFlowRateDefinedBy: StandardRho
        -> stdmfr: 1e-6[m^3/s]
        -> mnst: 0.0224136[m^3/mol]
        -> Pst: 1[atm]
        -> Tst: 273.15[K]
        -> sccmmfr: 100
        -> Dbnd: 1.0[m]
        -> Mf_src: userdef
        -> Mf: 0
        -> Mn_src: userdef
        -> Mn: 0.032[kg/mol]
        -> RANSVarOption: SpecifyTurbulentLengthScaleAndIntensity
        -> RANSAnisotropyOption: IsotropicTurbulence
        -> IT: spf.IT_init
        -> LT: spf.LT_init
        -> Uref: 1[m/s]
        -> k0: spf.k0_init
        -> ep0: spf.ep0_init
        -> om0: spf.om0_init
        -> nutilde0: 3*spf.nu
        -> zeta0: spf.zeta0_init
        -> zeta0_aniso: 2/3
        -> alpha0: spf.alpha0_init
        -> IT_list: user_defined
        -> LT_list: user_defined
        -> multipleInlets: 0
        -> AverageTotalPressure: 1
        -> PressureType: StaticPressure
        -> xi0: 1
        -> includeSyntheticTurb: 0
        -> gamma0: spf.gamma0_init
        -> N: 100
        -> useRandomSeed: 1
        -> phys.randomSeed: 113013
        -> RSMAnisotropyOption: IsotropicTurbulence
        -> R0: (2/3)*spf.k0_init
        -> a0: 0
      * out1 (OutletBoundary)
        -> BoundaryCondition: Pressure
        -> p0: 0
        -> NormalFlow: 0
        -> SuppressBackflow: 1
        -> ComponentWise: NormalOutflowVelocity
        -> U0out: 0
        -> u0: [0, 0, 0]
        -> LaminarOutflowOption: Uav
        -> Uav: 0
        -> V0: 0
        -> Lexit: 1
        -> p0_exit: 0
        -> Dzexit: 1
        -> ConstrainEndPointsToZero: 0
        -> FullyDevelopedFlowOption: Uav
        -> Uavfdf: 0
        -> V0fdf: 0
        -> p0avfdf: 0
        -> Dzfdf: 1
        -> multipleInlets: 0
        -> AverageTotalPressure: 1
        -> PressureType: StaticPressure
        -> xi0: 1
        -> MassFlowType: MassFlowRate
        -> mfr: 1e-5[kg/s]
        -> StandardFlowRateDefinedBy: StandardRho
        -> stdmfr: 1e-6[m^3/s]
        -> mnst: 0.0224136[m^3/mol]
        -> Pst: 1[atm]
        -> Tst: 273.15[K]
        -> sccmmfr: 100
        -> Dbnd: 1.0[m]
        -> Mf_src: userdef
        -> Mf: 0
        -> Mn_src: userdef
        -> Mn: 0.032[kg/mol]
    - tds (DilutedSpecies)
      * cdm1 (Fluid)
        -> u_src: root.comp1.u
        -> u: [0, 0, 0]
        -> DiffusionCoefficientSource: mat
        -> DiffusionMaterialList: mat1
        -> DH_mat: userdef
        -> DH: 9.3e-9[m^2/s]
        -> DOH_mat: userdef
        -> DOH: 5.3e-9[m^2/s]
        -> um: [1e-13[s*mol/kg], 1e-13[s*mol/kg], 1e-13[s*mol/kg], 1e-13[s*mol/kg], 1e-13[s*mol/kg], 1e-13[s*mol/kg], 1e-13[s*mol/kg], 1e-13[s*mol/kg], 1e-13[s*mol/kg]]
        -> umH: 1e-13[s*mol/kg]
        -> umOH: 1e-13[s*mol/kg]
        -> MobilityModel: NernstEinstein
        -> V_src: userdef
        -> V: 0
        -> D_PT_mat: userdef
        -> D_PT: D_PT+Ds
        -> Dchem_PT_src: userdef
        -> Dchem_PT: 1e-9[m^2/s]
        -> D_rp_mat: userdef
        -> D_rp: D_RP+Ds
        -> Dchem_rp_src: userdef
        -> Dchem_rp: 1e-9[m^2/s]
        -> D_ap_mat: userdef
        -> D_ap: D_AP+Ds
        -> Dchem_ap_src: userdef
        -> Dchem_ap: 1e-9[m^2/s]
        -> D_apr_mat: userdef
        -> D_apr: D_APR
        -> Dchem_apr_src: userdef
        -> Dchem_apr: 1e-9[m^2/s]
        -> D_aps_mat: userdef
        -> D_aps: D_APS
        -> Dchem_aps_src: userdef
        -> Dchem_aps: 1e-9[m^2/s]
        -> D_th_mat: userdef
        -> D_th: D_T+Ds
        -> Dchem_th_src: userdef
        -> Dchem_th: 1e-9[m^2/s]
        -> D_at_mat: userdef
        -> D_at: D_AT+Ds
        -> Dchem_at_src: userdef
        -> Dchem_at: 1e-9[m^2/s]
        -> D_fg_mat: userdef
        -> D_fg: D_FG
        -> Dchem_fg_src: userdef
        -> Dchem_fg: 1e-9[m^2/s]
        -> D_fi_mat: userdef
        -> D_fi: D_FI
        -> Dchem_fi_src: userdef
        -> Dchem_fi: 1e-9[m^2/s]
        -> phic_src: userdef
        -> phic: 0
        -> phid_src: userdef
        -> phid: 0
        -> uc_src: userdef
        -> uc: [0, 0, 0]
        -> ud_src: userdef
        -> ud: [0, 0, 0]
        -> Dm_rp: 1e-9[m^2/s]
        -> Dm_ap: 1e-9[m^2/s]
        -> Dm_apr: 1e-9[m^2/s]
        -> Dm_aps: 1e-9[m^2/s]
        -> Dm_PT: 1e-9[m^2/s]
        -> Dm_th: 1e-9[m^2/s]
        -> Dm_at: 1e-9[m^2/s]
        -> Dm_fg: 1e-9[m^2/s]
        -> Dm_fi: 1e-9[m^2/s]
        -> epsilonr_mat: userdef
        -> epsilonr: 80
      * nflx1 (NoFlux)
        -> IncludeConvection: 1
      * init1 (init)
        -> initc: [c_RP0, c_AP0, c_adp0, c_txa20, c_pT0, c_T0, c_aT0, c_Fg0, 0]
      * fl1 (FluxBoundary)
        -> IncludeConvection: 1
        -> FluxType: GeneralInwardFlux
        -> species: [1, 1, 1, 1, 1, 1, 0, 0, 0]
        -> J0: [(-if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_rs*RP, 0)-if(spf.sr<lss, Sat(M)*k_rs*RP, 0))*step2t(t), (-((if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_as*AP, 0)+if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Mas/M_inf*k_as*AP, 0)))-((if(spf.sr<lss, Sat(M)*k_as*AP, 0)+if(spf.sr<lss, Mas/M_inf*k_as*AP, 0))))*step2t(t), ((if(d(spf.sr,x)<sgt, lambda*(L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_rs*RP, 0)+if(spf.sr<lss,lambda*Sat(M)*k_rs*RP, 0)))*step2t(t), Mat*s_t*step2t(t), -beta*(phi_at*Mat)*PT*step2t(t), beta*(phi_at*Mat)*PT*step2t(t), 0, 0, 0]
        -> kc: [0, 0, 0, 0, 0, 0, 0, 0, 0]
        -> cb: [0, 0, 0, 0, 0, 0, 0, 0, 0]
      * reac1 (Reactions)
        -> R_PT_src: userdef
        -> R_PT: -beta*PT*(phi_rt*RP+phi_at*AP)
        -> ReactingVolumeType: TotalVolume
        -> R_rp_src: userdef
        -> R_rp: -k_pa(kpa_chem(Omega(T,APR,APS)),kpa_mech(spf.sr))*RP
        -> R_ap_src: userdef
        -> R_ap: k_pa(kpa_chem(Omega(T,APR,APS)),kpa_mech(spf.sr))*RP
        -> R_apr_src: userdef
        -> R_apr: lambda*k_pa(kpa_chem(Omega(T,APR,APS)),kpa_mech(spf.sr))*RP
        -> R_aps_src: userdef
        -> R_aps: s_t*AP-k_i*APS
        -> R_th_src: userdef
        -> R_th: PT*(phi_at*AP+phi_rt*RP)*beta-Gamma(T,AT)*T
        -> R_at_src: userdef
        -> R_at: -Gamma(AT,T)*T
        -> R_fg_src: userdef
        -> R_fg: -(kfi*FG*T)/(kmfi+FG)
        -> R_fi_src: userdef
        -> R_fi: (kfi*FG*T)/(kmfi+FG)
        -> chemTag: userdef
        -> Chemistry: 0
        -> ReactionHeatSource: userdef
        -> Q: 0
      * in1 (Inflow)
        -> c0: [c_RP0, c_AP0, 0, 0, c_pT0, 0, c_aT0, c_Fg0, 0]
        -> BoundaryConditionType: FluxDanckwerts
      * out1 (Outflow)
    - tds2 (DilutedSpecies)
      * cdm1 (Fluid)
        -> u_src: userdef
        -> u: [0, 0, 0]
        -> DiffusionCoefficientSource: mat
        -> DiffusionMaterialList: mat1
        -> DH_mat: userdef
        -> DH: 9.3e-9[m^2/s]
        -> DOH_mat: userdef
        -> DOH: 5.3e-9[m^2/s]
        -> um: [1e-13[s*mol/kg], 1e-13[s*mol/kg], 1e-13[s*mol/kg]]
        -> umH: 1e-13[s*mol/kg]
        -> umOH: 1e-13[s*mol/kg]
        -> MobilityModel: NernstEinstein
        -> V_src: userdef
        -> V: 0
        -> D_M_mat: userdef
        -> D_M: 0
        -> Dchem_M_src: userdef
        -> Dchem_M: 1e-9[m^2/s]
        -> D_Mas_mat: userdef
        -> D_Mas: 0
        -> Dchem_Mas_src: userdef
        -> Dchem_Mas: 1e-9[m^2/s]
        -> D_Mat_mat: userdef
        -> D_Mat: 0
        -> Dchem_Mat_src: userdef
        -> Dchem_Mat: 1e-9[m^2/s]
        -> phic_src: userdef
        -> phic: 0
        -> phid_src: userdef
        -> phid: 0
        -> uc_src: userdef
        -> uc: [0, 0, 0]
        -> ud_src: userdef
        -> ud: [0, 0, 0]
        -> Dm_M: 1e-9[m^2/s]
        -> Dm_Mas: 1e-9[m^2/s]
        -> Dm_Mat: 1e-9[m^2/s]
        -> epsilonr_mat: userdef
        -> epsilonr: 80
      * nflx1 (NoFlux)
        -> IncludeConvection: 0
      * init1 (init)
        -> initc: [0, 0, 0]
      * srf1 (SurfaceReactionsFlux)
        -> J0_M_src: userdef
        -> J0_M: Da*((if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_rs*RP, 0)+if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_as*AP, 0)+if(spf.sr<lss,Sat(M)*(k_rs*RP+k_as*AP),0)))*step2t(t)
        -> J0_Mas_src: userdef
        -> J0_Mas: Da*((if(d(spf.sr,x)<sgt,(L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_rs*RP, 0)+if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_as*AP, 0)+if(spf.sr<lss,Sat(M)*(k_rs*RP+k_as*AP),0)))*step2t(t)
        -> J0_Mat_src: userdef
        -> J0_Mat: Da*((if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_rs*RP, 0)+if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Sat(M)*k_as*AP, 0)+if(d(spf.sr,x)<sgt, (L/gamma_m)*abs(d(spf.sr,x))*Mas/M_inf*k_aa*AP, 0)+if(spf.sr<lss, Sat(M)*k_rs*RP+Sat(M)*k_as*AP+(Mas/M_inf)*k_aa*AP,0)))*step2t(t)

=== FUNCTIONS (Component comp1) ===
Count: 1
- Relation Between Wall Distance in Viscous Units and Kolmogorov-Scaled Wall Distance [Interpolation] :: lsupstarInv()
    * [Table contains 201 rows - showing first 5]
    * ['0', '0.00']
    * ['1.9607843137254897', '3.290628731016117']
    * ['4.4043748152527336', '6.84177822635119']
    * ['7.330771504581732', '11.382534075636396']
    * ['10.739974381712484', '17.885928261298574']
    * ...

Component: mcomp1
  Physics interfaces: 0

=== FUNCTIONS (Component mcomp1) ===
Count: 0

=== APPLIED BOUNDARY CONDITIONS / TARGET SELECTIONS ===

Component: comp1
  Physics: spf (LaminarFlow)
    - fp1 (FluidProperties) :: Fluid Properties 1
      applies on -> selection (geom1, dim=): dim 2: [1]
    - init1 (init) :: Initial Values 1
      applies on -> selection (geom1, dim=): dim 2: [1]
    - wallbc1 (WallBC) :: WallFluidBC
      applies on -> selection (geom1, dim=): dim 1: [2, 3]
    - inl1 (InletBoundary) :: InletFluidBC
      applies on -> named selection: box1
    - out1 (OutletBoundary) :: OutletFluidBC
      applies on -> named selection: box2
  Physics: tds (DilutedSpecies)
    - cdm1 (Fluid) :: Transport Properties 1
      applies on -> selection (geom1, dim=): dim 2: [1]
    - nflx1 (NoFlux) :: No Flux 1
      applies on -> no explicit entities reported
    - init1 (init) :: InitialValues_9spec
      applies on -> selection (geom1, dim=): dim 2: [1]
    - fl1 (FluxBoundary) :: WallFlux_9spec
      applies on -> named selection: dif1
    - reac1 (Reactions) :: Reactions_9spec
      applies on -> selection (geom1, dim=): dim 2: [1]
    - in1 (Inflow) :: InletFlux_9spec
      applies on -> named selection: box1
    - out1 (Outflow) :: ExitFlux_9spec
      applies on -> named selection: box2
  Physics: tds2 (DilutedSpecies)
    - cdm1 (Fluid) :: Transport Properties 1
      applies on -> selection (geom1, dim=): dim 2: [1]
    - nflx1 (NoFlux) :: NoFlux_InletAndOutlet_3spec
      applies on -> selection (geom1, dim=): dim 1: [1, 4]
    - init1 (init) :: InitialValues_3spec
      applies on -> selection (geom1, dim=): dim 2: [1]
    - srf1 (SurfaceReactionsFlux) :: wall_surface_reactions_3spec
      applies on -> named selection: dif1

Component: mcomp1

=== FULL EQUATION FORMS ===

  Physics: spf (LaminarFlow)
    Feature: fp1 (FluidProperties)
      - mu0: 0.56*(mu2(FI)+mu1(Mat))
      - minput_magneticfluxdensity_src: userdef
      - minput_magneticfluxdensity: [0, 0, 0]
      - minput_frequency_src: userdef
      - minput_frequency: root.freq
      - mu_init_app: spf.mu0
      - mu0_mat: userdef
      - mu0cy_mat: from_mat
      - mu0cy: 1e-3[Pa*s]
      - mu0cw_mat: from_mat
      - mu0cw: 1e-3[Pa*s]
      - mu0e_mat: from_mat
      - mu0e: 1e-3[Pa*s]
      - mu0c_mat: from_mat
      - mu0c: 1e-3[Pa*s]
    Feature: init1 (init)
      - u_init: [0, 0, 0]
      - p_init: 0
      - nutilde_init: spf.nutildeinit
      - G_init: spf.G0
      - k_init: spf.kinit
      - ep_init: spf.epinit
      - om_init: spf.omInit
      - yPlus_init: spf.yPlusinit
      - uPlus_init: spf.uPlusinit
      - zeta_init: 2/3
      - alpha_init: 1
      - gamma_init: 1
      - R_init: (2/3)*spf.kinit
    Feature: inl1 (InletBoundary)
      - U0in: inlet(x)
      - u0: [0, 0, 0]
      - p0: 800*stept(t)
      - p0_entr: 0
      - p0avfdf: 0
      - ep0: spf.ep0_init
    Feature: out1 (OutletBoundary)
      - p0: 0
      - U0out: 0
      - u0: [0, 0, 0]
      - p0_exit: 0
      - p0avfdf: 0

  Physics: tds (DilutedSpecies)
    Feature: cdm1 (Fluid)
      - DiffusionCoefficientSource: mat
      - DiffusionMaterialList: mat1
    Feature: init1 (init)
      - initc: [c_RP0, c_AP0, c_adp0, c_txa20, c_pT0, c_T0, c_aT0, c_Fg0, 0]
    Feature: fl1 (FluxBoundary)
      - FluxType: GeneralInwardFlux
    Feature: reac1 (Reactions)
      - ReactionHeatSource: userdef
    Feature: in1 (Inflow)
      - c0: [c_RP0, c_AP0, 0, 0, c_pT0, 0, c_aT0, c_Fg0, 0]

  Physics: tds2 (DilutedSpecies)
    Feature: cdm1 (Fluid)
      - DiffusionCoefficientSource: mat
      - DiffusionMaterialList: mat1
    Feature: init1 (init)
      - initc: [0, 0, 0]

=== MATERIALS CONTENT ===

  - mat1: Material 1 (Common)
      * Property group: def (Basic)
          - density: rho_b
          - dynamicviscosity: mu_b*(mu1(Mat)+mu2(FI))
      * Property group: Carreau (Carreau model)
          - mu0: 0.56*(mu2(FI)+mu1(Mat))
          - mu_inf: mu_b*(mu2(FI)+mu1(Mat))
          - lam_car: 3.313
          - n_car: 0.3568

=== PARAMETERS (global) ===
Count: 57
- rho_b = 1.106[g/cm^3] [None] :: blood density
- mu_b = 3.5e-2[g/(cm*s)] [None] :: blood viscosity
- D_RP = 1.58e-9[cm^2/s] [None] :: Diffusion coef activated platelets
- D_AP = 1.58e-9[cm^2/s] [None] :: Diffusion coef activated platelets
- D_APR = 2.57e-6[cm^2/s] [None] :: Diffusion coef adp agonist
- D_APS = 2.14e-6[cm^2/s] [None] :: Diffusion coef TxA2 agonist
- D_PT = 3.32e-7[cm^2/s] [None] :: Diffusion coef prothrombin agonist
- D_T = 4.16e-7[cm^2/s] [None] :: Diffusion coef thrombin agonist
- D_AT = 3.49e-7[cm^2/s] [None] :: Diffusion coef antithrombin
- c_RP0 = 2.5e8[plt/ml] [None] :: Initial RP concentration
- c_AP0 = 0.05*c_RP0 [plt/ml] [None] :: Initial AP concentration
- c_adp0 = 0[uM] [None] :: Initial adp concentration
- c_txa20 = 0[uM] [None] :: Initial TxA2 concentration
- c_pT0 = 1.2[uM] [None] :: Initial prothrombin concentration
- c_T0 = 0[U/ml] [None] :: Initial thrombin concentration
- c_aT0 = 2.84 [uM] [None] :: Initial antithrombin concentration
- w_adp = 1 [None] :: act weight adp
- w_txa2 = 1 [None] :: act weight thromboxane
- w_t = 1 [None] :: act weight thrombin
- APRcrit = 2[uM] [None] :: adp concentration for activation
- APScrit = 0.6[uM] [None] :: thromboxane concentration for activation
- Tcrit = 0.0005[uM] [None] :: thrombin concentration for activation
- t_act = 1[s] [None] :: activation time
- lambda = 2.4e-8[nmol/plt] [None] :: released adp/plt AP
- s_t = 9.5e-12[nmol/(s*plt)] [None] :: rate of synthesis of txa2
- k_i = 0.0161[1/s] [None] :: rate of txa2 inactivation
- phi_at = 3.69e-9[U/(plt*s*uM)] [None] :: thrombin generation rate at the surface of AP
- phi_rt = 6.5e-10[U/(plt*s*uM)] [None] :: thrombin generation rate at the surface of RP
- c_H = 0.25[uM] [None] :: heparin concentration [U/ml]
- k_1t = 13.33[1/s] [None] :: rate constant for aT
- K_at = .1[uM] [None] :: dissociation constant heparin-T
- K_T = 3.5e-2[uM] [None] :: dissociation constant heparin-aT
- M_inf = 7e6[plt/cm^2] [None] :: Total deposition capacity
- k_rs = 0.0037[cm/s] [None] :: adhesion rate
- k_as = 0.045 [cm/s] [None] :: adhesion rate
- k_aa = 0.045[cm/s] [None] :: adhesion rate
- beta = 9.11e-3[nmol/U] [None] :: Conversion factor for thrombin concentration
- beta2 = 9.11e-3 [None]
- tau_max = 2000[1/s] [None] :: Max shear rate
- dRBC = 5.5e-4[cm] [None] :: Keller diff coef
- tacc = 1e-3 [None] :: accuracy tolerance
- theta = 1 [None]
- Lb = 0.0035 [None]
- shear_crit = 10000[1/s] [None]
- Vplt = 4.18*10^-12[cm^3] [None]
- omega = 2*pi[1/s] [None]
- gamma_m = 150 [1/s] [None]
- lss = 25 [1/s] [None] :: low shear rate treshold
- sgt = -750 [1/(cm*s)] [None]
- L = 0.075[cm] [None]
- Da = 0.0001 [s/cm^2] [None]
- kmfi = 3.16 [uM] [None] :: Rate constant fibrin reaction
- kfi = 59 [1/s] [None] :: Reaction rate fibrinogen
- D_FI = 2.47*10^(-7) [cm^2/s] [None] :: Fibrin diffusion coefficient
- D_FG = 3.10*10^(-7) [cm^2/s] [None] :: Fibrinogen diffusion coefficient
- c_Fg0 = 7[uM] [None] :: Initial fibrinogen concentration
- Re_target = 450 [None]

=== VARIABLES (global + component) ===
Global variable groups: 4
  - var1: 1 vars (Scope: Global)
    * Ds = 0.18*dRBC^2*(tau_max)/4
  - var2: 8 vars (Scope: Global)
    * T = if(th<0,eps,th)
    * FI = if(fi<0,eps,fi)
    * RP = if(rp<0,eps,rp)
    * AP = if(ap<0,eps,ap)
    * APR = if(apr<0,eps,apr)
    * APS = if(aps<0,eps,aps)
    * FG = if(fg<0,eps,fg)
    * AT = if(at<0,eps,at)
  - var6: 3 vars (Scope: Global)
    * is_inlet = sel1(x,y) :: inlet identifier
    * is_outlet = sel2(x,y) :: outlet identifier
    * is_wall = sel3(x,y) :: wall identifier
  - var7: 2 vars (Scope: Global)
    * L_inlet = intop1(1)
    * U_inlet = (Re_target * mu_b) / (rho_b * L_inlet)
Component comp1 variable groups: 3
  - var2: 8 vars (Scope: empty selection)
    * T = if(th<0,eps,th)
    * FI = if(fi<0,eps,fi)
    * RP = if(rp<0,eps,rp)
    * AP = if(ap<0,eps,ap)
    * APR = if(apr<0,eps,apr)
    * APS = if(aps<0,eps,aps)
    * FG = if(fg<0,eps,fg)
    * AT = if(at<0,eps,at)
  - var6: 3 vars (Global/Entire Geometry)
    * is_inlet = sel1(x,y) :: inlet identifier
    * is_outlet = sel2(x,y) :: outlet identifier
    * is_wall = sel3(x,y) :: wall identifier
  - var7: 2 vars (Scope: empty selection)
    * L_inlet = intop1(1)
    * U_inlet = (Re_target * mu_b) / (rho_b * L_inlet)
Component mcomp1 variable groups: 0

=== NAMED/EXPLICIT SELECTIONS ===
Count: 3 (Ignored 13 auto-imports)
- box1: inlet [Box]
  entities -> dim 1: [1]
- box2: outlet [Box]
  entities -> dim 1: [4]
- dif1: wall [Difference]
  entities -> dim 1: [2, 3]

=== FUNCTIONS (Global) ===
Count: 13
- Analytic 1 [Analytic] :: Omega(T, APR, APS) = (APS/APScrit)+(APR/APRcrit)+(T/Tcrit)
- Analytic 2 [Analytic] :: kpa_chem(Omega) = if(Omega<500,(Omega/t_act)*Act_step(Omega),500)
- Analytic 3 [Analytic] :: Gamma(T, AT) = (k_1t*c_H*AT)/(K_at*K_T+T*K_at+AT*T)
- Analytic 4 [Analytic] :: Sat(M) = 1-M/M_inf
- Step 1 [Step] :: Act_step() -> Step at 1 (from 0 to 1)
- Step 2 [Step] :: stept() -> Step at 5.5 (from 0 to 1)
- Analytic 6 [Analytic] :: kpa_mech(spf.sr) = if(spf.sr>shear_crit, spf.sr/shear_crit, 0)
- Analytic 7 [Analytic] :: k_pa(kpa_chem, kpa_mech) = kpa_chem+kpa_mech
- smooth_thrombus [Piecewise] :: pw()
- viscosity platelets [Step] :: mu1() -> Step at 20000000 (from 1 to 80)
- Step 4 [Step] :: step2t() -> Step at 12 (from 0 to 1)
- viscosity fibrin [Step] :: mu2() -> Step at 0.6 (from 0 to 80)
- Relation Between Wall Distance in Viscous Units and Kolmogorov-Scaled Wall Distance [Interpolation] :: lsupstarInv()
    * [Table contains 201 rows - showing first 5]
    * ['0', '0.00']
    * ['1.9607843137254897', '3.290628731016117']
    * ['4.4043748152527336', '6.84177822635119']
    * ['7.330771504581732', '11.382534075636396']
    * ['10.739974381712484', '17.885928261298574']
    * ...
```

