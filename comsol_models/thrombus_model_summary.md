Contents

[1. Global Definitions](#cs8967057)

[1.1. Parameters](#cs6959583)

[1.2. Variables](#cs1721442)

[1.3. Functions](#cs7011622)

[1.4. Mesh Parts](#cs7471129)

[1.5. Shared Properties](#cs9089853)

[2. Component 1](#cs3564560)

[2.1. Definitions](#cs3017648)

[2.2. Geometry 1](#cs9550289)

[2.3. Materials](#cs8299633)

[2.4. Laminar Flow](#cs5803002)

[2.5. Transport of Diluted Species 9](#cs2924354)

[2.6. Transport of Diluted Species 2](#cs5090672)

[2.7. Multiphysics](#cs9050570)

[2.8. Mesh 1](#cs5223475)

[3. Study 1](#cs4406831)

[3.1. Time Dependent](#cs4939190)

[3.2. Solver Configurations](#cs4972792)

[4. Studio 2](#cs8098323)

[4.1. Transitorio](#cs6282868)

[4.2. Solver Configurations](#cs7654450)

[5. Results](#cs4837308)

[5.1. Data Sets](#cs3309613)

[5.2. Derived Values](#cs8099237)

[5.3. Tables](#cs8674428)

[5.4. Plot Groups](#cs4726330)

# Global Definitions

|||
|-|-|
|Date|Mar 12, 2026, 3:23:54 PM|

Global settings

|||
|-|-|
|Name|Phase2.mph|
|Path|C:\\Users\\pgssy\\LadHyX\_ml\_cfd\_thrombus\_predictions\\comsol\_models\\phase2.mph|
|Version|COMSOL Multiphysics 6.3 (Build: 420)|
|Unit system|None\[unit\_system]|

Used products

||
|-|
|COMSOL Multiphysics|
|CFD Module|

Computer information

|||
|-|-|
|CPU|Intel64 Family 6 Model 170 Stepping 4, 11 cores, 31.51 GB RAM|
|Operating system|Windows 11|

## Parameters

Parameters 1

|**Name**|**Expression**|**Value**|**Description**|
|-|-|-|-|
|H|0.16\[cm]|0.16|half channel height|
|rho\_b|1.106\[g/cm^3]|1.106|blood density|
|mu\_b|3.5e-2\[g/(cm\*s)]|0.035|blood viscosity|
|Q\_b|1.25e-2\[cm^3/s]|0.0125|blood flow rate|
|D\_RP|1.58e-9\[cm^2/s]|1.58E−9|Diffusion coef activated platelets|
|D\_AP|1.58e-9\[cm^2/s]|1.58E−9|Diffusion coef activated platelets|
|D\_APR|2.57e-6\[cm^2/s]|2.57E−6|Diffusion coef adp agonist|
|D\_APS|2.14e-6\[cm^2/s]|2.14E−6|Diffusion coef TxA2 agonist|
|D\_PT|3.32e-7\[cm^2/s]|3.32E−7|Diffusion coef prothrombin agonist|
|D\_T|4.16e-7\[cm^2/s]|4.16E−7|Diffusion coef thrombin agonist|
|D\_AT|3.49e-7\[cm^2/s]|3.49E−7|Diffusion coef antithrombin|
|c\_RP0|2.5e8\[plt/ml]|2.5E8|Initial RP concentration|
|c\_AP0|0.05\*c\_RP0 \[plt/ml]|1.25E7|Initial AP concentration|
|c\_adp0|0\[uM]|0|Initial adp concentration|
|c\_txa20|0\[uM]|0|Initial TxA2 concentration|
|c\_pT0|1.2\[uM]|1.2|Initial prothrombin concentration|
|c\_T0|0\[U/ml]|0|Initial thrombin concentration|
|c\_aT0|2.84 \[uM]|2.84|Initial antithrombin concentration|
|w\_adp|1|1|act weight adp|
|w\_txa2|1|1|act weight thromboxane|
|w\_t|1|1|act weight thrombin|
|APRcrit|2\[uM]|2|adp concentration for activation|
|APScrit|0.6\[uM]|0.6|thromboxane concentration for activation|
|Tcrit|0.0005\[uM]|5E−4|thrombin concentration for activation|
|t\_act|1\[s]|1|activation time|
|lambda|2.4e-8\[nmol/plt]|2.4E−8|released adp/plt AP|
|s\_t|9.5e-12\[nmol/(s\*plt)]|9.5E−12|rate of synthesis of txa2|
|k\_i|0.0161\[1/s]|0.0161|rate of txa2 inactivation|
|phi\_at|3.69e-9\[U/(plt\*s\*uM)]|3.69E−9|thrombin generation rate at the surface of AP|
|phi\_rt|6.5e-10\[U/(plt\*s\*uM)]|6.5E−10|thrombin generation rate at the surface of RP|
|c\_H|0.25\[uM]|0.25|heparin concentration \[U/ml]|
|k\_1t|13.33\[1/s]|13.33|rate constant for aT|
|K\_at|.1\[uM]|0.1|dissociation constant heparin-T|
|K\_T|3.5e-2\[uM]|0.035|dissociation constant heparin-aT|
|M\_inf|7e6\[plt/cm^2]|7E6|Total deposition capacity|
|k\_rs|0.0037\[cm/s]|0.0037|adhesion rate|
|k\_as|0.045 \[cm/s]|0.045|adhesion rate|
|k\_aa|0.045\[cm/s]|0.045|adhesion rate|
|beta|9.11e-3\[nmol/U]|0.00911|Conversion factor for thrombin concentration|
|beta2|9.11e-3|0.00911||
|tau\_max|2000\[1/s]|2000|Max shear rate|
|dRBC|5.5e-4\[cm]|5.5E−4|Keller diff coef|
|tacc|1e-3|0.001|accuracy tolerance|
|theta|1|1||
|Lb|0.0035|0.0035||
|A|5.6|5.6||
|shear\_crit|10000\[1/s]|10000||
|Vplt|4.18\*10^ - 12\[cm^3]|4.18E−12||
|omega|2\*pi\[1/s]|6.2832||
|gamma\_m|150 \[1/s]|150||
|lss|25 \[1/s]|25|low shear rate treshold|
|sgt|-750 \[1/(cm\*s)]|−750||
|L|0.075\[cm]|0.075||
|Da|0.0001 \[s/cm^2]|1E−4||
|Ld|5 \[cm]|5||
|kmfi|3.16 \[uM]|3.16|Rate constant fibrin reaction|
|kfi|59 \[1/s]|59|Reaction rate fibrinogen|
|D\_FI|2.47\*10^(-7) \[cm^2/s]|2.47E−7|Fibrin diffusion coefficient|
|D\_FG|3.10\*10^(-7) \[cm^2/s]|3.1E−7|Fibrinogen diffusion coefficient|
|c\_Fg0|7\[uM]|7|Initial fibrinogen concentration|
|ra|0.25\[cm]|0.25|radius of the aneurysm|
|area|(2\*pi\*H\*Ld + (4\*pi\*(ra)^2)) \[cm^2]|5.8119||
|U\_inlet|(Re\_target \* mu\_b) / (rho\_b \* D\_eff)|7.1676|inlet avg velocity (FD)|
|Re\_target|450|450||
|D\_eff|1.986780366 \[cm]|1.9868||
|U\_max|1.5 \* U\_inlet|10.751||

## Variables

### Variables 1

Selection

|||
|-|-|
|Geometric entity level|Entire model|

|**Name**|**Expression**|**Description**|
|-|-|-|
|U|(tau\_max\*H/2)\*(1 - ((H - y)^2/H^2))||
|Ds|0.18\*dRBC^2\*(tau\_max)/4||

## Functions

### Analytic 1

|||
|-|-|
|Function name|Omega|
|Function type|Analytic|

Definition

|**Description**|**Value**|
|-|-|
|Expression|(APS/APScrit) + (APR/APRcrit) + (T/Tcrit)|
|Arguments|{T, APR, APS}|

Units

|**Argument**|**Unit**|
|-|-|
|T||
|APR||
|APS||

### Analytic 2

|||
|-|-|
|Function name|kpa\_chem|
|Function type|Analytic|

Definition

|**Description**|**Value**|
|-|-|
|Expression|if(Omega<500, (Omega/t\_act)\*Act\_step(Omega), 500)|
|Arguments|Omega|

Units

|**Argument**|**Unit**|
|-|-|
|Omega||

### Analytic 3

|||
|-|-|
|Function name|Gamma|
|Function type|Analytic|

Definition

|**Description**|**Value**|
|-|-|
|Expression|(k\_1t\*c\_H\*AT)/(K\_at\*K\_T + T\*K\_at + AT\*T)|
|Arguments|{T, AT}|

Units

|**Argument**|**Unit**|
|-|-|
|T||
|AT||

### Analytic 4

|||
|-|-|
|Function name|Sat|
|Function type|Analytic|

Definition

|**Description**|**Value**|
|-|-|
|Expression|1 - M/M\_inf|
|Arguments|M|

Units

|**Argument**|**Unit**|
|-|-|
|M||

### Step 1

|||
|-|-|
|Function name|Act\_step|
|Function type|Step|

Parameters

|**Description**|**Value**|
|-|-|
|Location|1|
|From|0|
|To|1|

### Step 2

|||
|-|-|
|Function name|stept|
|Function type|Step|

Parameters

|**Description**|**Value**|
|-|-|
|Location|5.5|
|From|0|
|To|1|

Smoothing

|**Description**|**Value**|
|-|-|
|Size of transition zone|10|

### Analytic 6

|||
|-|-|
|Function name|kpa\_mech|
|Function type|Analytic|

Definition

|**Description**|**Value**|
|-|-|
|Expression|if(spf.sr>shear\_crit, spf.sr/shear\_crit, 0)|
|Arguments|spf.sr|

Units

|**Argument**|**Unit**|
|-|-|
|spf.sr||

### Analytic 7

|||
|-|-|
|Function name|k\_pa|
|Function type|Analytic|

Definition

|**Description**|**Value**|
|-|-|
|Expression|kpa\_chem + kpa\_mech|
|Arguments|{kpa\_chem, kpa\_mech}|

Units

|**Argument**|**Unit**|
|-|-|
|kpa\_chem||
|kpa\_mech||

### smooth\_thrombus

|||
|-|-|
|Function name|pw|
|Function type|Piecewise|

Definition

|**Description**|**Value**|
|-|-|
|Argument|y|
|Extrapolation|Constant|
|Smoothing|Continuous second derivative|
|Size of transition zone|0.45|

Definition

|**Start**|**End**|**Function**|
|-|-|-|
|0|2.2|0|
|2.2|2.8|1|
|2.8|5|0|

### viscosity platelets

|||
|-|-|
|Function name|mu1|
|Function type|Step|

Parameters

|**Description**|**Value**|
|-|-|
|Location|2E7|
|From|1|
|To|80|

Smoothing

|**Description**|**Value**|
|-|-|
|Size of transition zone|7E6|

### Step 4

|||
|-|-|
|Function name|step2t|
|Function type|Step|

Parameters

|**Description**|**Value**|
|-|-|
|Location|12|
|From|0|
|To|1|

Smoothing

|**Description**|**Value**|
|-|-|
|Size of transition zone|2.5|

### viscosity fibrin

|||
|-|-|
|Function name|mu2|
|Function type|Step|

Parameters

|**Description**|**Value**|
|-|-|
|Location|0.6|
|From|0|
|To|80|

Smoothing

|**Description**|**Value**|
|-|-|
|Size of transition zone|0.01|

### inlet

|||
|-|-|
|Function name|inlet|
|Function type|Interpolation|

Units

|**Function**|**Unit**|
|-|-|
|inlet|cm/s|

Units

|**Argument**|**Unit**|
|-|-|
|t|cm|

## Mesh Parts

### Parte della mesh 1

Mesh statistics

|**Description**|**Value**|
|-|-|
|Status|Complete mesh with second-order elements|
|Mesh vertices|63559|
|Triangles|31458|
|Edge elements|642|
|Vertex elements|46|
|Number of elements|31458|
|Minimum element quality|0.6012|
|Average element quality|0.9131|
|Element area ratio|0.13464|
|Mesh area|22.22|

#### Importa 1 (imp1)

Information

|**Description**|**Value**|
|-|-|
|Source|NASTRAN file|
|Last build time|< 1 second|
|Built with|COMSOL 6.0.0.318 (win64), Dec 2, 2025, 4:00:07 PM|

Settings

|**Description**|**Value**|
|-|-|
|Filename|C:\\Users\\marko\\Downloads\\VirtualSurgery\_TPPOD1\_mesh.nas|

Settings

|**Name**|**Source in file**|
|-|-|
|ID 1 Importa 1|PID 1|

Information

|**NASTRAN entry**|**Number**|
|-|-|
|GRID|63559|
|CTRIA6|31458|
|PSHELL|1|

#### Finalizza (fin)

Information

|**Description**|**Value**|
|-|-|
|Last build time|< 1 second|
|Built with|COMSOL 6.0.0.318 (win64), Dec 2, 2025, 4:00:07 PM|

## Shared Properties

### Common model inputs 1

|||
|-|-|
|Tag|cminpt|

# Component 1

|||
|-|-|
|Date|Jan 31, 2019, 10:50:04 AM|

Settings

|**Description**|**Value**|
|-|-|
|Unit system|Same as global system (None)|
|Geometry shape function|Automatic|
|Avoid inverted elements by curving interior domain elements|Off|

Spatial frame coordinates

|**First**|**Second**|**Third**|
|-|-|-|
|x|y|z|

Material frame coordinates

|**First**|**Second**|**Third**|
|-|-|-|
|X|Y|Z|

Geometry frame coordinates

|**First**|**Second**|**Third**|
|-|-|-|
|Xg|Yg|Zg|

Mesh frame coordinates

|**First**|**Second**|**Third**|
|-|-|-|
|Xm|Ym|Zm|

## Definitions

### Variables

#### Variables 2

Selection

|||
|-|-|
|Geometric entity level|Entire model|

|**Name**|**Expression**|**Description**|
|-|-|-|
|T|if(th<0, eps, th)||
|FI|if(fi<0, eps, fi)||
|RP|if(rp<0, eps, rp)||
|AP|if(ap<0, eps, ap)||
|APR|if(apr<0, eps, apr)||
|APS|if(aps<0, eps, aps)||
|FG|if(fg<0, eps, fg)||
|AT|if(at<0, eps, at)||

#### domain

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: Domain 1|

|**Name**|**Expression**|**Description**|
|-|-|-|
|is\_inlet|sel1(x, y)|inlet identifier|
|is\_outlet|sel2(x, y)|outlet identifier|
|is\_wall|sel3(x, y)|wall identifier|

### Selections

#### inlet

|**Selection type**|
|-|
|Explicit|

|**Selection**|
|-|
|Boundary 5|

#### outlet

|**Selection type**|
|-|
|Explicit|

|**Selection**|
|-|
|Boundary 8|

#### wall

|**Selection type**|
|-|
|Explicit|

|**Selection**|
|-|
|Boundaries 1–4, 6–7|

### Coordinate Systems

#### Boundary System 1

|||
|-|-|
|Coordinate system type|Boundary system|
|Tag|sys1|

Coordinate names

|**First**|**Second**|**Third**|
|-|-|-|
|t1|n|to|

### Shared Properties

#### Model Input 1

|||
|-|-|
|Tag|minpt1|

Definition

|**Description**|**Value**|
|-|-|
|Variable name|minput.T|

## Geometry 1

Geometry statistics

|**Description**|**Value**|
|-|-|
|Space dimension|2|
|Number of domains|1|
|Number of boundaries|8|
|Number of vertices|8|

### Import 1 (imp1)

Source

|**Description**|**Value**|
|-|-|
|Source|Mesh|
|Mesh|[Parte della mesh 1 {mpart1}](#cs2362460)|

Selections of resulting entities

|**Description**|**Value**|
|-|-|
|Resulting objects selection|On|
|Show in physics|All levels|

Information

|**Description**|**Value**|
|-|-|
|Build message|Imported 1 solid object from Parte della mesh 1.|

### Rotate 1 (rot1)

Input objects

|**Description**|**Value**|
|-|-|
|Input objects|geom1, Geometry geom1: Object: imp1|

Rotation

|**Description**|**Value**|
|-|-|
|Angle|-90|

Center of rotation

|**Description**|**Value**|
|-|-|
|Position|{0, 0}|

### Move 1 (mov1)

Input objects

|**Description**|**Value**|
|-|-|
|Input objects|geom1, Geometry geom1: Object: rot1|

Displacement

|**Description**|**Value**|
|-|-|
|Specify|Displacement vector|
|x|0|
|y|-3.975|

### Form Union (fin)

Information

|**Description**|**Value**|
|-|-|
|Build message|Formed union of 1 solid object. Union has 1 domain, 46 boundaries, and 46 vertices.|

### Ignora vertici 1 (igv1)

Input objects

|**Description**|**Value**|
|-|-|
|Vertices to ignore|geom1, Geometry geom1: Dimension 0: Object: fin: Points 3–11, 13–16, 18–46|

### Partizione lati 1 (pare1)

Input objects

|**Description**|**Value**|
|-|-|
|Edges to partition|geom1, Geometry geom1: Dimension 1: Object: igv1: Boundary 2|

Position

|**Relative arc length parameters**|
|-|
|0.3|
|0.05|

### Partizione lati 2 (pare2)

Input objects

|**Description**|**Value**|
|-|-|
|Edges to partition|geom1, Geometry geom1: Dimension 1: Object: pare1: Boundary 1|

Position

|**Relative arc length parameters**|
|-|
|0.755|
|0.95|

## Materials

### Material 1

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

Basic

|**Description**|**Value**|
|-|-|
|Density|rho\_b|
|Dynamic viscosity|mu\_b\*(mu1(Mat) + mu2(FI))|

Carreau model

|**Description**|**Value**|
|-|-|
|Zero shear rate viscosity|0.56\*(mu2(FI) + mu1(Mat))|
|Infinite shear rate viscosity|mu\_b\*(mu2(FI) + mu1(Mat))|
|Relaxation time|3.313|
|Power index|0.3568|
|Apparent viscosity||

## Laminar Flow

Used products

||
|-|
|COMSOL Multiphysics|
|CFD Module|

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: Domain 1|

Equations

### Interface Settings

#### Discretization

Settings

|**Description**|**Value**|
|-|-|
|Discretization of fluids|P2 + P1|

Settings

|**Description**|**Value**|
|-|-|
|Equation form|Study controlled|

#### Physical Model

Settings

|**Description**|**Value**|
|-|-|
|Neglect inertial term (Stokes flow)|Off|
|Compressibility|Incompressible flow|
|Use shallow channel approximation|Off|
|Enable porous media domains|Off|
|Include gravity|Off|
|Reference temperature|User defined|
|Reference temperature|293.15\[K]|
|Reference pressure level|1\[atm]|

#### Turbulence

Settings

|**Description**|**Value**|
|-|-|
|Turbulence model type|None|

### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|spf.Tref|model.input.Tref|Reference temperature|Global|Meta|
|spf.dz|1|Thickness|Domain 1||
|spf.pref|1\[atm]|Reference pressure level|Domain 1||
|spf.pA|p+spf.pref|Absolute pressure|Domain 1||
|spf.hasWF|0|Help variable|Boundaries 1–8||
|spf.dt\_CFL|1/max(spf.maxop(sqrt(emetric\_spatial(u-d(x,TIME),v-d(y,TIME)))),eps)|Time step, CFL=1|Global||
|spf.CFL\_number|timestep/spf.dt\_CFL|CFL number|Global||
|spf.Qvd\_tot|spf.intop(spf.Qvd\*spf.dz)|Total viscous dissipation|Global||
|spf.K\_stressx|spf.K\_stress\_tensorxx\*spf.nxmesh+spf.K\_stress\_tensorxy\*spf.nymesh+spf.K\_stress\_tensorxz\*spf.nzmesh|Viscous stress, exterior boundaries, x-component|Boundaries 1–8||
|spf.K\_stressy|spf.K\_stress\_tensoryx\*spf.nxmesh+spf.K\_stress\_tensoryy\*spf.nymesh+spf.K\_stress\_tensoryz\*spf.nzmesh|Viscous stress, exterior boundaries, y-component|Boundaries 1–8||
|spf.K\_stressz|spf.K\_stress\_tensorzx\*spf.nxmesh+spf.K\_stress\_tensorzy\*spf.nymesh+spf.K\_stress\_tensorzz\*spf.nzmesh|Viscous stress, exterior boundaries, z-component|Boundaries 1–8||
|spf.T\_stressx|spf.T\_stress\_tensorxx\*spf.nxmesh+spf.T\_stress\_tensorxy\*spf.nymesh+spf.T\_stress\_tensorxz\*spf.nzmesh|Total traction, exterior boundaries, x-component|Boundaries 1–8||
|spf.T\_stressy|spf.T\_stress\_tensoryx\*spf.nxmesh+spf.T\_stress\_tensoryy\*spf.nymesh+spf.T\_stress\_tensoryz\*spf.nzmesh|Total traction, exterior boundaries, y-component|Boundaries 1–8||
|spf.T\_stressz|spf.T\_stress\_tensorzx\*spf.nxmesh+spf.T\_stress\_tensorzy\*spf.nymesh+spf.T\_stress\_tensorzz\*spf.nzmesh|Total traction, exterior boundaries, z-component|Boundaries 1–8||
|spf.K\_stress\_dx|down(spf.K\_stress\_tensorxx)\*spf.dnxmesh+down(spf.K\_stress\_tensorxy)\*spf.dnymesh+down(spf.K\_stress\_tensorxz)\*spf.dnzmesh|Viscous stress, interior boundaries, downside, x-component|Boundaries 1–8||
|spf.K\_stress\_dy|down(spf.K\_stress\_tensoryx)\*spf.dnxmesh+down(spf.K\_stress\_tensoryy)\*spf.dnymesh+down(spf.K\_stress\_tensoryz)\*spf.dnzmesh|Viscous stress, interior boundaries, downside, y-component|Boundaries 1–8||
|spf.K\_stress\_dz|down(spf.K\_stress\_tensorzx)\*spf.dnxmesh+down(spf.K\_stress\_tensorzy)\*spf.dnymesh+down(spf.K\_stress\_tensorzz)\*spf.dnzmesh|Viscous stress, interior boundaries, downside, z-component|Boundaries 1–8||
|spf.T\_stress\_dx|down(spf.T\_stress\_tensorxx)\*spf.dnxmesh+down(spf.T\_stress\_tensorxy)\*spf.dnymesh+down(spf.T\_stress\_tensorxz)\*spf.dnzmesh|Total traction, interior boundaries, downside, x-component|Boundaries 1–8||
|spf.T\_stress\_dy|down(spf.T\_stress\_tensoryx)\*spf.dnxmesh+down(spf.T\_stress\_tensoryy)\*spf.dnymesh+down(spf.T\_stress\_tensoryz)\*spf.dnzmesh|Total traction, interior boundaries, downside, y-component|Boundaries 1–8||
|spf.T\_stress\_dz|down(spf.T\_stress\_tensorzx)\*spf.dnxmesh+down(spf.T\_stress\_tensorzy)\*spf.dnymesh+down(spf.T\_stress\_tensorzz)\*spf.dnzmesh|Total traction, interior boundaries, downside, z-component|Boundaries 1–8||
|spf.T\_tracx|spf.T\_stressx|Total applied traction, exterior boundaries, x-component|Boundaries 1–8||
|spf.T\_tracy|spf.T\_stressy|Total applied traction, exterior boundaries, y-component|Boundaries 1–8||
|spf.T\_tracz|spf.T\_stressz|Total applied traction, exterior boundaries, z-component|Boundaries 1–8||
|spf.T\_trac\_dx|spf.T\_stress\_dx|Total applied traction, downside boundaries, x-component|Boundaries 1–8||
|spf.T\_trac\_dy|spf.T\_stress\_dy|Total applied traction, downside boundaries, y-component|Boundaries 1–8||
|spf.T\_trac\_dz|spf.T\_stress\_dz|Total applied traction, downside boundaries, z-component|Boundaries 1–8||
|spf.delid|0.25|Tuning parameter|Domain 1||
|spf.usePseudoTimeStepping|isrunningpseudotimestepping|Help variable|Global||
|spf.localCFLvalue|1.3^min(niterCMP,9)+if(niterCMP>=25,9\*1.3^min(-25+niterCMP,9),0)+if(niterCMP>=45,90\*1.3^min(-45+niterCMP,9),0)|Local CFL number|Domain 1||
|spf.locCFL|max(CFLCMP,sqrt(eps))|Local CFL number|Global||
|spf.geometryLengthScale|1.6928717123902206|Geometry length scale|Domain 1||
|spf.time\_step\_inv|max(sqrt(emetric\_spatial(u,v)\*2^if(gmg\_level<2,0,-1+gmg\_level)^2),spf.nu/spf.geometryLengthScale^2)|Inverse time step|Domain 1||
|spf.tsti|nojac(spf.time\_step\_inv/spf.locCFL)|Help variable|Domain 1||
|spf.nx|dnx|Normal vector, x-component|Boundaries 1–8||
|spf.ny|dny|Normal vector, y-component|Boundaries 1–8||
|spf.nz|0|Normal vector, z-component|Boundaries 1–8||
|spf.nxmesh|dnxmesh|Normal vector, x-component|Boundaries 1–8||
|spf.nymesh|dnymesh|Normal vector, y-component|Boundaries 1–8||
|spf.nzmesh|0|Normal vector, z-component|Boundaries 1–8||

### Fluid Properties 1

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

Equations

#### Fluid Properties

Settings

|**Description**|**Value**|
|-|-|
|Density|User defined|
|Density|rho\_b|
|Constitutive relation|Inelastic non - Newtonian|
|Inelastic model|Carreau|
|Zero shear rate viscosity|User defined|
|Zero shear rate viscosity|0.56\*(mu2(FI) + mu1(Mat))|
|Infinite shear rate viscosity|User defined|
|Infinite shear rate viscosity|mu\_b\*(mu2(FI) + mu1(Mat))|
|Relaxation time|User defined|
|Relaxation time|3.313|
|Power index|User defined|
|Power index|0.3568|

#### Thermal Effects

Settings

|**Description**|**Value**|
|-|-|
|Thermal function|None|

#### Model Input

Settings

|**Description**|**Value**|
|-|-|
|Temperature|Common model input|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|spf.nu|spf.mu/spf.rho|Kinematic viscosity|Domain 1||
|spf.mu0|0.56\*(mu2(FI)+mu1(Mat))|Zero shear rate viscosity|Domain 1||
|spf.mu\_inf|mu\_b\*(mu2(FI)+mu1(Mat))|Infinite shear rate viscosity|Domain 1||
|spf.lam\_car|3.313|Relaxation time|Domain 1||
|spf.n\_car|0.3568|Power index|Domain 1||
|spf.rho|material.rho|Density|Domain 1|Meta|
|spf.Trho|spf.fp1.minput\_temperature|Temperature for density evaluation|Domain 1||
|spf.prho|spf.fp1.minput\_pressure|Pressure for the evaluation of density|Domain 1||
|spf.rhoref|subst(material.rho,minput.T,spf.Tref,minput.pA,spf.pref)|Reference density|Domain 1|Meta|
|spf.mumat|nojac(spf.mu\_app)|Dynamic viscosity|Domain 1||
|spf.srijxx|ux|Strain rate tensor, xx-component|Domain 1||
|spf.srijyx|0.5\*(vx+uy)|Strain rate tensor, yx-component|Domain 1||
|spf.srijzx|0|Strain rate tensor, zx-component|Domain 1||
|spf.srijxy|0.5\*(uy+vx)|Strain rate tensor, xy-component|Domain 1||
|spf.srijyy|vy|Strain rate tensor, yy-component|Domain 1||
|spf.srijzy|0|Strain rate tensor, zy-component|Domain 1||
|spf.srijxz|0|Strain rate tensor, xz-component|Domain 1||
|spf.srijyz|0|Strain rate tensor, yz-component|Domain 1||
|spf.srijzz|0|Strain rate tensor, zz-component|Domain 1||
|spf.rrijxx|0|Rotation rate tensor, xx-component|Domain 1||
|spf.rrijyx|0.5\*(vx-uy)|Rotation rate tensor, yx-component|Domain 1||
|spf.rrijzx|0|Rotation rate tensor, zx-component|Domain 1||
|spf.rrijxy|0.5\*(uy-vx)|Rotation rate tensor, xy-component|Domain 1||
|spf.rrijyy|0|Rotation rate tensor, yy-component|Domain 1||
|spf.rrijzy|0|Rotation rate tensor, zy-component|Domain 1||
|spf.rrijxz|0|Rotation rate tensor, xz-component|Domain 1||
|spf.rrijyz|0|Rotation rate tensor, yz-component|Domain 1||
|spf.rrijzz|0|Rotation rate tensor, zz-component|Domain 1||
|spf.sr|sqrt(2\*spf.srijxx^2+2\*spf.srijxy^2+2\*spf.srijxz^2+2\*spf.srijyx^2+2\*spf.srijyy^2+2\*spf.srijyz^2+2\*spf.srijzx^2+2\*spf.srijzy^2+2\*spf.srijzz^2+eps)|Shear rate|Domain 1||
|spf.rr|sqrt(2\*spf.rrijxx^2+2\*spf.rrijxy^2+2\*spf.rrijxz^2+2\*spf.rrijyx^2+2\*spf.rrijyy^2+2\*spf.rrijyz^2+2\*spf.rrijzx^2+2\*spf.rrijzy^2+2\*spf.rrijzz^2+eps)|Rotation rate|Domain 1||
|spf.divu|ux+vy|Divergence of velocity field|Domain 1||
|spf.Fx|0|Volume force, x-component|Domain 1|+ operation|
|spf.Fy|0|Volume force, y-component|Domain 1|+ operation|
|spf.Fz|0|Volume force, z-component|Domain 1|+ operation|
|spf.U|sqrt(u^2+v^2)|Velocity magnitude|Domain 1||
|spf.vorticityx|0|Vorticity field, x-component|Domain 1||
|spf.vorticityy|0|Vorticity field, y-component|Domain 1||
|spf.vorticityz|vx-uy|Vorticity field, z-component|Domain 1||
|spf.vort\_magn|sqrt(spf.vorticityx^2+spf.vorticityy^2+spf.vorticityz^2)|Vorticity magnitude|Domain 1||
|spf.cellRe|0.25\*spf.rho\*sqrt(emetric\_spatial(u-d(x,TIME),v-d(y,TIME))/emetric2\_spatial)/spf.mu|Cell Reynolds number|Domain 1||
|spf.betaT|0|Isothermal compressibility coefficient|Domain 1||
|spf.Qm|0|Source term|Domain 1|+ operation|
|spf.Fgtotx|0|Gravity force, x-component|Domain 1|+ operation|
|spf.Fgtoty|0|Gravity force, y-component|Domain 1|+ operation|
|spf.Fgtotz|0|Gravity force, z-component|Domain 1|+ operation|
|spf.Qm\_aco|0|Acoustic mass source|Domain 1||
|spf.F\_acox|0|Acoustic volume force, x-component|Domain 1||
|spf.F\_acoy|0|Acoustic volume force, y-component|Domain 1||
|spf.F\_acoz|0|Acoustic volume force, z-component|Domain 1||
|spf.gamma\_sr|sqrt(2\*spf.srijxx^2+2\*spf.srijxy^2+2\*spf.srijxz^2+2\*spf.srijyx^2+2\*spf.srijyy^2+2\*spf.srijyz^2+2\*spf.srijzx^2+2\*spf.srijzy^2+2\*spf.srijzz^2+eps)|Shear rate|Domain 1||
|spf.mu\_eff|spf.mu+spf.muT|Effective dynamic viscosity|Domain 1||
|spf.muT|0|Turbulent dynamic viscosity|Domain 1|+ operation|
|spf.T\_stress\_tensorxx|spf.K\_stress\_tensorxx-p|Total stress tensor, xx-component|Domain 1|+ operation|
|spf.T\_stress\_tensoryx|spf.K\_stress\_tensoryx|Total stress tensor, yx-component|Domain 1|+ operation|
|spf.T\_stress\_tensorzx|spf.K\_stress\_tensorzx|Total stress tensor, zx-component|Domain 1|+ operation|
|spf.T\_stress\_tensorxy|spf.K\_stress\_tensorxy|Total stress tensor, xy-component|Domain 1|+ operation|
|spf.T\_stress\_tensoryy|spf.K\_stress\_tensoryy-p|Total stress tensor, yy-component|Domain 1|+ operation|
|spf.T\_stress\_tensorzy|spf.K\_stress\_tensorzy|Total stress tensor, zy-component|Domain 1|+ operation|
|spf.T\_stress\_tensorxz|spf.K\_stress\_tensorxz|Total stress tensor, xz-component|Domain 1|+ operation|
|spf.T\_stress\_tensoryz|spf.K\_stress\_tensoryz|Total stress tensor, yz-component|Domain 1|+ operation|
|spf.T\_stress\_tensorzz|spf.K\_stress\_tensorzz-p|Total stress tensor, zz-component|Domain 1|+ operation|
|spf.K\_stress\_tensorxx|2\*spf.mu\_eff\*ux|Viscous stress tensor, xx-component|Domain 1|+ operation|
|spf.K\_stress\_tensoryx|spf.mu\_eff\*(vx+uy)|Viscous stress tensor, yx-component|Domain 1|+ operation|
|spf.K\_stress\_tensorzx|0|Viscous stress tensor, zx-component|Domain 1|+ operation|
|spf.K\_stress\_tensorxy|spf.mu\_eff\*(uy+vx)|Viscous stress tensor, xy-component|Domain 1|+ operation|
|spf.K\_stress\_tensoryy|2\*spf.mu\_eff\*vy|Viscous stress tensor, yy-component|Domain 1|+ operation|
|spf.K\_stress\_tensorzy|0|Viscous stress tensor, zy-component|Domain 1|+ operation|
|spf.K\_stress\_tensorxz|0|Viscous stress tensor, xz-component|Domain 1|+ operation|
|spf.K\_stress\_tensoryz|0|Viscous stress tensor, yz-component|Domain 1|+ operation|
|spf.K\_stress\_tensorzz|0|Viscous stress tensor, zz-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testxx|2\*spf.mu\_eff\*test(ux)|Viscous stress tensor test, xx-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testyx|spf.mu\_eff\*(test(vx)+test(uy))|Viscous stress tensor test, yx-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testzx|0|Viscous stress tensor test, zx-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testxy|spf.mu\_eff\*(test(uy)+test(vx))|Viscous stress tensor test, xy-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testyy|2\*spf.mu\_eff\*test(vy)|Viscous stress tensor test, yy-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testzy|0|Viscous stress tensor test, zy-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testxz|0|Viscous stress tensor test, xz-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testyz|0|Viscous stress tensor test, yz-component|Domain 1|+ operation|
|spf.K\_stress\_tensor\_testzz|0|Viscous stress tensor test, zz-component|Domain 1|+ operation|
|spf.upwind\_helpx|u-d(x,TIME)|Upwind term, x-component|Domain 1|+ operation|
|spf.upwind\_helpy|v-d(y,TIME)|Upwind term, y-component|Domain 1|+ operation|
|spf.upwind\_helpz|0|Upwind term, z-component|Domain 1|+ operation|
|spf.continuityEquation|spf.rho\*spf.divu-spf.Qm|Continuity equation|Domain 1||
|spf.contCoeff|spf.rho|Help variable|Domain 1||
|spf.mu\_app|spf.mu\_inf+(spf.mu0-spf.mu\_inf)\*(1+(spf.lam\_car\*spf.sr)^2)^(0.5\*(-1+spf.n\_car))|Apparent viscosity|Domain 1||
|spf.mu|nojac(spf.mu\_app)|Dynamic viscosity|Domain 1||
|spf.Qvd|spf.K\_stress\_tensorxx\*ux+spf.K\_stress\_tensorxy\*uy+spf.K\_stress\_tensoryx\*vx+spf.K\_stress\_tensoryy\*vy|Viscous dissipation|Domain 1|+ operation|
|spf.isodiffns|-spf.delid\*h\_spatial\*sqrt((spf.rho\*u)^2+(spf.rho\*v)^2+eps)\*(ux\*test(ux)+uy\*test(uy)+vx\*test(vx)+vy\*test(vy))|Isotropic diffusion|Domain 1||
|spf.epsilon\_p|1|Porosity|Domain 1||
|spf.epsilon\_p\_pos|max(1,sqrt(eps))|Positive porosity|Domain 1||
|spf.Fst\_tensorxx|0|Surface tension force, xx-component|Domain 1|+ operation|
|spf.Fst\_tensoryx|0|Surface tension force, yx-component|Domain 1|+ operation|
|spf.Fst\_tensorzx|0|Surface tension force, zx-component|Domain 1|+ operation|
|spf.Fst\_tensorxy|0|Surface tension force, xy-component|Domain 1|+ operation|
|spf.Fst\_tensoryy|0|Surface tension force, yy-component|Domain 1|+ operation|
|spf.Fst\_tensorzy|0|Surface tension force, zy-component|Domain 1|+ operation|
|spf.Fst\_tensorxz|0|Surface tension force, xz-component|Domain 1|+ operation|
|spf.Fst\_tensoryz|0|Surface tension force, yz-component|Domain 1|+ operation|
|spf.Fst\_tensorzz|0|Surface tension force, zz-component|Domain 1|+ operation|
|spf.res\_u|spf.rho\*ut\*spf.switch\_NS+px+spf.rho\*u\*ux+spf.rho\*v\*uy-(d(2\*ux,x)+d(uy+vx,y))\*spf.mu-spf.Fx|Equation residual|Domain 1||
|spf.res\_v|spf.rho\*vt\*spf.switch\_NS+spf.rho\*u\*vx+py+spf.rho\*v\*vy-(d(vx+uy,x)+d(2\*vy,y))\*spf.mu-spf.Fy|Equation residual|Domain 1||
|spf.res\_p|spf.rho\*spf.divu-spf.Qm|Pressure equation residual|Domain 1||

#### Shape functions

|**Name**|**Shape function**|**Description**|**Shape frame**|**Selection**|
|-|-|-|-|-|
|u|Lagrange (Quadratic)|Velocity field, x-component|Spatial|Domain 1|
|v|Lagrange (Quadratic)|Velocity field, y-component|Spatial|Domain 1|
|u|Lagrange (Quadratic)|Velocity field, x-component|Spatial|Domain 1|
|v|Lagrange (Quadratic)|Velocity field, y-component|Spatial|Domain 1|
|p|Lagrange (Linear)|Pressure|Spatial|Domain 1|

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|spf.rho\*(-ut\*test(u)-vt\*test(v))|4|Spatial|Domain 1|
|(p-spf.K\_stress\_tensorxx)\*test(ux)-spf.K\_stress\_tensorxy\*test(uy)-spf.K\_stress\_tensoryx\*test(vx)+(p-spf.K\_stress\_tensoryy)\*test(vy)|4|Spatial|Domain 1|
|spf.Fx\*test(u)+spf.Fy\*test(v)|4|Spatial|Domain 1|
|spf.rho\*(-(d(u,x)\*u+d(u,y)\*v)\*test(u)-(d(v,x)\*u+d(v,y)\*v)\*test(v))|4|Spatial|Domain 1|
|-spf.continuityEquation\*test(p)|4|Spatial|Domain 1|
|spf.isodiffns|4|Spatial|Domain 1|
|spf.streamlinens|4|Spatial|Domain 1|
|spf.crosswindns|4|Spatial|Domain 1|

### Initial Values 1

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

#### Initial Values

Settings

|**Description**|**Value**|
|-|-|
|Velocity field, x-component|0|
|Velocity field, y-component|0|
|Velocity field, z-component|0|
|Pressure|0|

#### Coordinate System Selection

Settings

|**Description**|**Value**|
|-|-|
|Coordinate system|Global coordinate system|

Used products

COMSOL Multiphysics

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|spf.u\_initx|0|Velocity field, x-component|Domain 1||
|spf.u\_inity|0|Velocity field, y-component|Domain 1||
|spf.u\_initz|0|Velocity field, z-component|Domain 1||
|spf.p\_init|0|Pressure|Domain 1||

### Wall 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: All boundaries|

Equations

#### Boundary Condition

Settings

|**Description**|**Value**|
|-|-|
|Wall condition|No slip|

#### Wall Movement

Settings

|**Description**|**Value**|
|-|-|
|Translational velocity|Automatic from frame|
|Sliding wall|Off|

Used products

COMSOL Multiphysics

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|spf.ubndx|spf.utrx+spf.usx|Velocity at boundary, x-component|Boundaries 1–4, 6–7||
|spf.ubndy|spf.utry+spf.usy|Velocity at boundary, y-component|Boundaries 1–4, 6–7||
|spf.ubndz|spf.utrz+spf.usz|Velocity at boundary, z-component|Boundaries 1–4, 6–7||
|spf.usx|0|Velocity of sliding wall, x-component|Boundaries 1–4, 6–7||
|spf.usy|0|Velocity of sliding wall, y-component|Boundaries 1–4, 6–7||
|spf.usz|0|Velocity of sliding wall, z-component|Boundaries 1–4, 6–7||
|spf.utrx|0|Velocity of moving wall, x-component|Boundaries 1–4, 6–7||
|spf.utry|0|Velocity of moving wall, y-component|Boundaries 1–4, 6–7||
|spf.utrz|0|Velocity of moving wall, z-component|Boundaries 1–4, 6–7||
|spf.uLeakagex|0|Leakage velocity, x-component|Boundaries 1–4, 6–7|+ operation|
|spf.uLeakagey|0|Leakage velocity, y-component|Boundaries 1–4, 6–7|+ operation|
|spf.uLeakagez|0|Leakage velocity, z-component|Boundaries 1–4, 6–7|+ operation|
|spf.noSlipWall|1|Help variable|Boundaries 1–4, 6–7||

#### Constraints

|**Constraint**|**Constraint force**|**Shape function**|**Selection**|**Details**|
|-|-|-|-|-|
|-u+spf.ubndx+spf.uLeakagex|test(-u)|Lagrange (Quadratic)|Boundaries 1–4, 6–7|Elemental|
|-v+spf.ubndy+spf.uLeakagey|test(-v)|Lagrange (Quadratic)|Boundaries 1–4, 6–7|Elemental|
|spf.ubndz+spf.uLeakagez|0||Boundaries 1–4, 6–7|Elemental|

### Inlet 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundary 5|

Equations

#### Boundary Condition

Settings

|**Description**|**Value**|
|-|-|
|Boundary condition|Fully developed flow|
|Apply condition on each disjoint selection separately|Off|

#### Fully Developed Flow

Settings

|**Description**|**Value**|
|-|-|
|Fully developed flow option|Average velocity|
|Average velocity|U\_inlet|

Used products

COMSOL Multiphysics

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|spf.nu|spf.mu/spf.rho|Kinematic viscosity|Boundary 5||
|spf.KStressn\_avx|spf.K\_stress\_tensorxx\*spf.nxmesh+spf.K\_stress\_tensorxy\*spf.nymesh+spf.K\_stress\_tensorxz\*spf.nzmesh|Average viscous stress, x-component|Boundary 5||
|spf.KStressn\_avy|spf.K\_stress\_tensoryx\*spf.nxmesh+spf.K\_stress\_tensoryy\*spf.nymesh+spf.K\_stress\_tensoryz\*spf.nzmesh|Average viscous stress, y-component|Boundary 5||
|spf.KStressn\_avz|spf.K\_stress\_tensorzx\*spf.nxmesh+spf.K\_stress\_tensorzy\*spf.nymesh+spf.K\_stress\_tensorzz\*spf.nzmesh|Average viscous stress, z-component|Boundary 5||
|spf.KStressTestn\_avx|spf.K\_stress\_tensor\_testxx\*spf.nxmesh+spf.K\_stress\_tensor\_testxy\*spf.nymesh+spf.K\_stress\_tensor\_testxz\*spf.nzmesh|Average viscous stress, x-component|Boundary 5||
|spf.KStressTestn\_avy|spf.K\_stress\_tensor\_testyx\*spf.nxmesh+spf.K\_stress\_tensor\_testyy\*spf.nymesh+spf.K\_stress\_tensor\_testyz\*spf.nzmesh|Average viscous stress, y-component|Boundary 5||
|spf.KStressTestn\_avz|spf.K\_stress\_tensor\_testzx\*spf.nxmesh+spf.K\_stress\_tensor\_testzy\*spf.nymesh+spf.K\_stress\_tensor\_testzz\*spf.nzmesh|Average viscous stress, z-component|Boundary 5||
|spf.ujumpx|spf.ut\_herex-spf.ut\_therex|Velocity jump, x-component|Boundary 5||
|spf.ujumpy|spf.ut\_herey-spf.ut\_therey|Velocity jump, y-component|Boundary 5||
|spf.ujumpz|spf.ut\_herez-spf.ut\_therez|Velocity jump, z-component|Boundary 5||
|spf.meshVol|meshvol\_spatial||Boundary 5||
|spf.meshVolInt|down(meshvol\_spatial)|Volume of interior mesh element|Boundary 5||
|spf.c\_here|72\*nojac(down((spf.mu+spf.muT)/spf.epsilon\_p))\*spf.meshVol/spf.meshVolInt|Intermediate variable|Boundary 5||
|spf.sigma\_dg\_ns|4\*spf.c\_here||Boundary 5||
|spf.inl1.Uavfdf|U\_inlet|Average velocity|Global||
|spf.inl1.dz|spf.dz|Channel thickness|Boundary 5||
|spf.un\_here|u\*nojac(spf.nxmesh)+v\*nojac(spf.nymesh)|Intermediate variable|Boundary 5||
|spf.ut\_herex|u-spf.un\_here\*nojac(spf.nxmesh)|Intermediate variable, x-component|Boundary 5||
|spf.ut\_herey|v-spf.un\_here\*nojac(spf.nymesh)|Intermediate variable, y-component|Boundary 5||
|spf.ut\_herez|-spf.un\_here\*nojac(spf.nzmesh)|Intermediate variable, z-component|Boundary 5||
|spf.un\_there|0|Intermediate variable|Boundary 5||
|spf.ut\_therex|-spf.un\_there\*nojac(spf.nxmesh)|Intermediate variable, x-component|Boundary 5||
|spf.ut\_therey|-spf.un\_there\*nojac(spf.nymesh)|Intermediate variable, y-component|Boundary 5||
|spf.ut\_therez|-spf.un\_there\*nojac(spf.nzmesh)|Intermediate variable, z-component|Boundary 5||
|spf.unTestx|(test(u)\*spf.nxmesh+test(v)\*spf.nymesh)\*spf.nxmesh||Boundary 5||
|spf.unTesty|(test(u)\*spf.nxmesh+test(v)\*spf.nymesh)\*spf.nymesh||Boundary 5||
|spf.unTestz|(test(u)\*spf.nxmesh+test(v)\*spf.nymesh)\*spf.nzmesh||Boundary 5||
|spf.inl1.pHydroCompensation|0|Hydrostatic pressure|Global||
|spf.d|1|Length|Boundary 5||
|spf.inl1.L|10\*spf.inl1.intop(1)|Entrance length|Global||
|spf.inl1.side|5|Help variable|Point 1||
|spf.inl1.side|5|Help variable|Point 2||
|spf.inl1.side\_down|5|Help variable|Point 1||
|spf.inl1.side\_down|5|Help variable|Point 2||
|spf.inl1.volumeFlowRate|spf.inl1.intop((u\*spf.nxmesh+v\*spf.nymesh)\*spf.inl1.dz)|Outward volume flow rate across feature selection|Global||
|spf.inl1.massFlowRate|spf.inl1.intop(spf.rho\*(u\*spf.nxmesh+v\*spf.nymesh)\*spf.inl1.dz)|Outward mass flow rate across feature selection|Global||
|spf.inl1.pAverage|spf.inl1.aveop(p)|Pressure average over feature selection|Global||
|spf.inl1.Vinlfdf|-spf.inl1.volumeFlowRate|Boundary integral of velocity|Global||
|spf.inl1.Area|spf.inl1.intop(spf.d)|Boundary area|Global||

#### Shape functions

|**Name**|**Shape function**|**Description**|**Shape frame**|**Selection**|
|-|-|-|-|-|
|spf.inl1.Pinlfdf|ODE|Help ode variable for fully developed flow||Global|

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|spf.KStressn\_avx\*test(spf.ut\_herex)+spf.KStressn\_avy\*test(spf.ut\_herey)+spf.KStressn\_avz\*test(spf.ut\_herez)+spf.KStressTestn\_avx\*spf.ujumpx+spf.KStressTestn\_avy\*spf.ujumpy+spf.KStressTestn\_avz\*spf.ujumpz-spf.sigma\_dg\_ns\*spf.ujumpx\*test(spf.ut\_herex)-spf.sigma\_dg\_ns\*spf.ujumpy\*test(spf.ut\_herey)-spf.sigma\_dg\_ns\*spf.ujumpz\*test(spf.ut\_herez)|4|Spatial|Boundary 5|
|((-2\*spf.mu\_eff\*uTx/spf.epsilon\_p+0.5\*(p+spf.inl1.Pinlfdf))\*dtang(spf.unTestx,x)-spf.mu\_eff\*(uTy+vTx)\*dtang(spf.unTesty,x)/spf.epsilon\_p-spf.mu\_eff\*(uTy+vTx)\*dtang(spf.unTestx,y)/spf.epsilon\_p+(-2\*spf.mu\_eff\*vTy/spf.epsilon\_p+0.5\*(p+spf.inl1.Pinlfdf))\*dtang(spf.unTesty,y))\*spf.inl1.L-(spf.nxmesh\*test(u)+spf.nymesh\*test(v))\*spf.inl1.Pinlfdf|4|Spatial|Boundary 5|
|(spf.inl1.Vinlfdf-spf.inl1.Area\*spf.inl1.Uavfdf)\*test(spf.inl1.Pinlfdf)|4||Global|

#### Constraints

|**Constraint**|**Constraint force**|**Shape function**|**Selection**|**Details**|
|-|-|-|-|-|
|-u+spf.ubndx|test(-u)|Lagrange (Quadratic)|Points 1–2|Elemental|
|-v+spf.ubndy|test(-v)|Lagrange (Quadratic)|Points 1–2|Elemental|
|spf.ubndz|0||Points 1–2|Elemental|
|-spf.inl1.side\_up(u)+spf.ubndx|test(-spf.inl1.side\_up(u))|Lagrange (Quadratic)|No points|Elemental|
|-spf.inl1.side\_up(v)+spf.ubndy|test(-spf.inl1.side\_up(v))|Lagrange (Quadratic)|No points|Elemental|
|-spf.inl1.side\_up(0)+spf.ubndz|test(-spf.inl1.side\_up(0))||No points|Elemental|
|-spf.inl1.side\_down(u)+spf.ubndx|test(-spf.inl1.side\_down(u))|Lagrange (Quadratic)|No points|Elemental|
|-spf.inl1.side\_down(v)+spf.ubndy|test(-spf.inl1.side\_down(v))|Lagrange (Quadratic)|No points|Elemental|
|-spf.inl1.side\_down(0)+spf.ubndz|test(-spf.inl1.side\_down(0))||No points|Elemental|

### Outlet 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundary 8|

Equations

#### Boundary Condition

Settings

|**Description**|**Value**|
|-|-|
|Boundary condition|Pressure|

#### Pressure Conditions

Settings

|**Description**|**Value**|
|-|-|
|Pressure|Static|
|Pressure|0|
|Normal flow|Off|
|Suppress backflow|On|

Used products

COMSOL Multiphysics

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|spf.meshVol|meshvol\_spatial||Boundary 8||
|spf.meshVolInt|down(meshvol\_spatial)|Volume of interior mesh element|Boundary 8||
|spf.rhoFace|down(spf.rho)|Density face value|Boundary 8||
|spf.umxTnFace|spf.upwind\_helpx\*spf.nxmesh+spf.upwind\_helpy\*spf.nymesh+spf.upwind\_helpz\*spf.nzmesh|Relative velocity on face|Boundary 8||
|spf.p0|0|Pressure|Boundary 8||
|spf.out1.Uav|0|Average velocity|Global||
|spf.out1.Uavfdf|0|Average velocity|Global||
|spf.Dbnd|1\[m]|Channel thickness|Boundary 8||
|spf.out1.dz|spf.dz|Channel thickness|Boundary 8||
|spf.out1.Mflow|spf.out1.massFlowRate|Mass flow|Global||
|spf.f0|spf.p0+spf.uNormal\*(spf.backflowPenaltyDiff-spf.backflowPenaltyConv)\*(spf.uNormal<0)|Normal stress|Boundary 8||
|spf.uNormal|u\*nojac(spf.nxmesh)+v\*nojac(spf.nymesh)|Normal velocity|Boundary 8||
|spf.out1.c\_here|288/spf.epsilon\_p|Intermediate variable|Boundary 8||
|spf.backflowPenaltyDiff|spf.out1.c\_here\*min((down(spf.mu)+spf.muT)\*spf.meshVol/spf.meshVolInt,down(spf.rho)\*abs(spf.uNormal)/down(spf.epsilon\_p))|Backflow penalty parameter, diffusive contribution|Boundary 8||
|spf.backflowPenaltyConv|spf.rhoFace\*spf.umxTnFace/spf.epsilon\_p^2|Backflow penalty parameter, convective contribution|Boundary 8||
|spf.out1.upwind\_ns|spf.backflowPenaltyConv\*spf.uNormal|Upwind term|Boundary 8||
|spf.out1.volumeFlowRate|spf.out1.intop((u\*spf.nxmesh+v\*spf.nymesh)\*spf.out1.dz)|Outward volume flow rate across feature selection|Global||
|spf.out1.massFlowRate|spf.out1.intop(spf.rho\*(u\*spf.nxmesh+v\*spf.nymesh)\*spf.out1.dz)|Outward mass flow rate across feature selection|Global||
|spf.out1.pAverage|spf.out1.aveop(p)|Pressure average over feature selection|Global||

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|-spf.f0\*(test(u)\*spf.nxmesh+test(v)\*spf.nymesh)|4|Spatial|Boundary 8|

## Transport of Diluted Species 9

Used products

||
|-|
|COMSOL Multiphysics|
|CFD Module|

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

Equations

### Interface Settings

#### Discretization

Settings

|**Description**|**Value**|
|-|-|
|Concentration|Linear|

Settings

|**Description**|**Value**|
|-|-|
|Equation form|Study controlled|

#### Out-of-Plane Thickness

Settings

|**Description**|**Value**|
|-|-|
|Out-of-plane thickness|1\[m]|

#### Species Activity

Settings

|**Description**|**Value**|
|-|-|
|Species activity|Ideal|

#### Inconsistent Stabilization

Settings

|**Description**|**Value**|
|-|-|
|Isotropic diffusion|On|
|Tuning parameter|0.25|

#### Transport Mechanisms

Settings

|**Description**|**Value**|
|-|-|
|Convection|On|
|Mass transfer in porous media|Off|

### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds.dz|1\[m]|Out-of-plane thickness|Global||
|tds.d|tds.dz|Out-of-plane geometry extension|Global||
|tds.f\_rp|1|Activity coefficient|Domain 1||
|tds.f\_ap|1|Activity coefficient|Domain 1||
|tds.f\_apr|1|Activity coefficient|Domain 1||
|tds.f\_aps|1|Activity coefficient|Domain 1||
|tds.f\_PT|1|Activity coefficient|Domain 1||
|tds.f\_th|1|Activity coefficient|Domain 1||
|tds.f\_at|1|Activity coefficient|Domain 1||
|tds.f\_fg|1|Activity coefficient|Domain 1||
|tds.f\_fi|1|Activity coefficient|Domain 1||
|tds.nx|dnx|Normal vector, x-component|Boundaries 1–8||
|tds.ny|dny|Normal vector, y-component|Boundaries 1–8||
|tds.nz|0|Normal vector, z-component|Boundaries 1–8||
|tds.nX|dnX|Normal vector, X-component|Boundaries 1–8||
|tds.nY|dnY|Normal vector, Y-component|Boundaries 1–8||
|tds.nZ|0|Normal vector, Z-component|Boundaries 1–8||
|tds.nXg|dnXg|Normal vector, Xg-component|Boundaries 1–8||
|tds.nYg|dnYg|Normal vector, Yg-component|Boundaries 1–8||
|tds.nZg|0|Normal vector, Zg-component|Boundaries 1–8||
|tds.nxmesh|dnxmesh|Normal vector (mesh), x-component|Boundaries 1–8||
|tds.nymesh|dnymesh|Normal vector (mesh), y-component|Boundaries 1–8||
|tds.nzmesh|0|Normal vector (mesh), z-component|Boundaries 1–8||
|tds.nxc|nxc/tds.ncLen|Normal vector, x-component|Boundaries 1–8||
|tds.nyc|nyc/tds.ncLen|Normal vector, y-component|Boundaries 1–8||
|tds.nzc|0|Normal vector, z-component|Boundaries 1–8||
|tds.ncLen|sqrt(nxc^2+nyc^2+eps)|Help variable|Boundaries 1–8||
|tds.cbf\_rp|0|Convective boundary flux|Boundaries 1–8||
|tds.u|0|Velocity field, x-component|Domain 1||
|tds.v|0|Velocity field, y-component|Domain 1||
|tds.w|0|Velocity field, z-component|Domain 1||
|tds.cbf\_ap|0|Convective boundary flux|Boundaries 1–8||
|tds.cbf\_apr|0|Convective boundary flux|Boundaries 1–8||
|tds.cbf\_aps|0|Convective boundary flux|Boundaries 1–8||
|tds.cbf\_PT|0|Convective boundary flux|Boundaries 1–8||
|tds.cbf\_th|0|Convective boundary flux|Boundaries 1–8||
|tds.cbf\_at|0|Convective boundary flux|Boundaries 1–8||
|tds.cbf\_fg|0|Convective boundary flux|Boundaries 1–8||
|tds.cbf\_fi|0|Convective boundary flux|Boundaries 1–8||
|tds.R\_rp|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_rp|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_rp|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_rp|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_rp|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_rp|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_rp|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_rp|rp|Species|Boundaries 1–8||
|tds.cVar\_rp|rp|Species|Points 1–8||
|tds.R\_ap|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_ap|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_ap|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_ap|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_ap|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_ap|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_ap|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_ap|ap|Species|Boundaries 1–8||
|tds.cVar\_ap|ap|Species|Points 1–8||
|tds.R\_apr|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_apr|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_apr|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_apr|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_apr|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_apr|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_apr|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_apr|apr|Species|Boundaries 1–8||
|tds.cVar\_apr|apr|Species|Points 1–8||
|tds.R\_aps|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_aps|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_aps|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_aps|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_aps|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_aps|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_aps|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_aps|aps|Species|Boundaries 1–8||
|tds.cVar\_aps|aps|Species|Points 1–8||
|tds.R\_PT|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_PT|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_PT|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_PT|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_PT|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_PT|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_PT|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_PT|PT|Species|Boundaries 1–8||
|tds.cVar\_PT|PT|Species|Points 1–8||
|tds.R\_th|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_th|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_th|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_th|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_th|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_th|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_th|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_th|th|Species|Boundaries 1–8||
|tds.cVar\_th|th|Species|Points 1–8||
|tds.R\_at|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_at|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_at|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_at|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_at|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_at|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_at|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_at|at|Species|Boundaries 1–8||
|tds.cVar\_at|at|Species|Points 1–8||
|tds.R\_fg|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_fg|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_fg|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_fg|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_fg|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_fg|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_fg|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_fg|fg|Species|Boundaries 1–8||
|tds.cVar\_fg|fg|Species|Points 1–8||
|tds.R\_fi|0|Total rate expression|Domain 1|+ operation|
|tds.cP\_fi|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds.cP\_fi|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds.KP\_fi|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds.KP\_fi|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds.Rads\_fi|0|Total adsorption rate|Domain 1|+ operation|
|tds.DiT\_fi|0|Turbulent diffusivity|Domain 1||
|tds.cVar\_fi|fi|Species|Boundaries 1–8||
|tds.cVar\_fi|fi|Species|Points 1–8||
|tds.poro|1|Porosity|Domain 1||
|tds.theta\_g|0|Gas volume fraction|Domain 1||
|tds.theta\_l|1|Liquid volume fraction|Domain 1||
|tds.theta|tds.poro|Mobile fluid volume fraction|Domain 1||

### Transport Properties 1

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

Equations

#### Convection

Settings

|**Description**|**Value**|
|-|-|
|Velocity field|Velocity field (spf)|

#### Diffusion

Settings

|**Description**|**Value**|
|-|-|
|Source|Material|
|Material|Material 1 (mat1) {mat1}|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_RP + Ds|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_AP + Ds|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_APR|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_APS|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_PT + Ds|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_T + Ds|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_AT + Ds|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_FG|
|Diffusion coefficient|User defined|
|Diffusion coefficient|D\_FI|

#### Coordinate System Selection

Settings

|**Description**|**Value**|
|-|-|
|Coordinate system|Global coordinate system|

#### Model Input

Settings

|**Description**|**Value**|
|-|-|
|Temperature|Common model input|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|domflux.rpx|tds.dflux\_rpx\*tds.d|Domain flux, x-component|Domain 1||
|domflux.rpy|tds.dflux\_rpy\*tds.d|Domain flux, y-component|Domain 1||
|domflux.apx|tds.dflux\_apx\*tds.d|Domain flux, x-component|Domain 1||
|domflux.apy|tds.dflux\_apy\*tds.d|Domain flux, y-component|Domain 1||
|domflux.aprx|tds.dflux\_aprx\*tds.d|Domain flux, x-component|Domain 1||
|domflux.apry|tds.dflux\_apry\*tds.d|Domain flux, y-component|Domain 1||
|domflux.apsx|tds.dflux\_apsx\*tds.d|Domain flux, x-component|Domain 1||
|domflux.apsy|tds.dflux\_apsy\*tds.d|Domain flux, y-component|Domain 1||
|domflux.PTx|tds.dflux\_PTx\*tds.d|Domain flux, x-component|Domain 1||
|domflux.PTy|tds.dflux\_PTy\*tds.d|Domain flux, y-component|Domain 1||
|domflux.thx|tds.dflux\_thx\*tds.d|Domain flux, x-component|Domain 1||
|domflux.thy|tds.dflux\_thy\*tds.d|Domain flux, y-component|Domain 1||
|domflux.atx|tds.dflux\_atx\*tds.d|Domain flux, x-component|Domain 1||
|domflux.aty|tds.dflux\_aty\*tds.d|Domain flux, y-component|Domain 1||
|domflux.fgx|tds.dflux\_fgx\*tds.d|Domain flux, x-component|Domain 1||
|domflux.fgy|tds.dflux\_fgy\*tds.d|Domain flux, y-component|Domain 1||
|domflux.fix|tds.dflux\_fix\*tds.d|Domain flux, x-component|Domain 1||
|domflux.fiy|tds.dflux\_fiy\*tds.d|Domain flux, y-component|Domain 1||
|tds.ndflux\_rp|tds.bndFlux\_rp|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_rp|tds.cflux\_rpx\*tds.nxc+tds.cflux\_rpy\*tds.nyc+tds.cflux\_rpz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_rp|tds.bndFlux\_rp+tds.cflux\_rpx\*tds.nxc+tds.cflux\_rpy\*tds.nyc+tds.cflux\_rpz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.ndflux\_ap|tds.bndFlux\_ap|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_ap|tds.cflux\_apx\*tds.nxc+tds.cflux\_apy\*tds.nyc+tds.cflux\_apz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_ap|tds.bndFlux\_ap+tds.cflux\_apx\*tds.nxc+tds.cflux\_apy\*tds.nyc+tds.cflux\_apz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.ndflux\_apr|tds.bndFlux\_apr|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_apr|tds.cflux\_aprx\*tds.nxc+tds.cflux\_apry\*tds.nyc+tds.cflux\_aprz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_apr|tds.bndFlux\_apr+tds.cflux\_aprx\*tds.nxc+tds.cflux\_apry\*tds.nyc+tds.cflux\_aprz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.ndflux\_aps|tds.bndFlux\_aps|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_aps|tds.cflux\_apsx\*tds.nxc+tds.cflux\_apsy\*tds.nyc+tds.cflux\_apsz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_aps|tds.bndFlux\_aps+tds.cflux\_apsx\*tds.nxc+tds.cflux\_apsy\*tds.nyc+tds.cflux\_apsz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.ndflux\_PT|tds.bndFlux\_PT|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_PT|tds.cflux\_PTx\*tds.nxc+tds.cflux\_PTy\*tds.nyc+tds.cflux\_PTz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_PT|tds.bndFlux\_PT+tds.cflux\_PTx\*tds.nxc+tds.cflux\_PTy\*tds.nyc+tds.cflux\_PTz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.ndflux\_th|tds.bndFlux\_th|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_th|tds.cflux\_thx\*tds.nxc+tds.cflux\_thy\*tds.nyc+tds.cflux\_thz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_th|tds.bndFlux\_th+tds.cflux\_thx\*tds.nxc+tds.cflux\_thy\*tds.nyc+tds.cflux\_thz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.ndflux\_at|tds.bndFlux\_at|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_at|tds.cflux\_atx\*tds.nxc+tds.cflux\_aty\*tds.nyc+tds.cflux\_atz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_at|tds.bndFlux\_at+tds.cflux\_atx\*tds.nxc+tds.cflux\_aty\*tds.nyc+tds.cflux\_atz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.ndflux\_fg|tds.bndFlux\_fg|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_fg|tds.cflux\_fgx\*tds.nxc+tds.cflux\_fgy\*tds.nyc+tds.cflux\_fgz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_fg|tds.bndFlux\_fg+tds.cflux\_fgx\*tds.nxc+tds.cflux\_fgy\*tds.nyc+tds.cflux\_fgz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.ndflux\_fi|tds.bndFlux\_fi|Normal diffusive flux|Boundaries 1–8||
|tds.ncflux\_fi|tds.cflux\_fix\*tds.nxc+tds.cflux\_fiy\*tds.nyc+tds.cflux\_fiz\*tds.nzc|Normal convective flux|Boundaries 1–8||
|tds.ntflux\_fi|tds.bndFlux\_fi+tds.cflux\_fix\*tds.nxc+tds.cflux\_fiy\*tds.nyc+tds.cflux\_fiz\*tds.nzc|Normal total flux|Boundaries 1–8||
|tds.u|model.input.u1|Velocity field, x-component|Domain 1|Meta|
|tds.v|model.input.u2|Velocity field, y-component|Domain 1|Meta|
|tds.w|model.input.u3|Velocity field, z-component|Domain 1|Meta|
|tds.DF\_rpxx|D\_RP+Ds|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_rpyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_rpzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_rpxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_rpyy|D\_RP+Ds|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_rpzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_rpxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_rpyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_rpzz|D\_RP+Ds|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_rpxx|tds.DF\_rpxx+tds.DiT\_rp|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_rpyx|tds.DF\_rpyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_rpzx|tds.DF\_rpzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_rpxy|tds.DF\_rpxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_rpyy|tds.DF\_rpyy+tds.DiT\_rp|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_rpzy|tds.DF\_rpzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_rpxz|tds.DF\_rpxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_rpyz|tds.DF\_rpyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_rpzz|tds.DF\_rpzz+tds.DiT\_rp|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.DF\_apxx|D\_AP+Ds|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_apyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_apzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_apxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_apyy|D\_AP+Ds|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_apzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_apxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_apyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_apzz|D\_AP+Ds|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_apxx|tds.DF\_apxx+tds.DiT\_ap|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_apyx|tds.DF\_apyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_apzx|tds.DF\_apzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_apxy|tds.DF\_apxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_apyy|tds.DF\_apyy+tds.DiT\_ap|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_apzy|tds.DF\_apzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_apxz|tds.DF\_apxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_apyz|tds.DF\_apyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_apzz|tds.DF\_apzz+tds.DiT\_ap|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.DF\_aprxx|D\_APR|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_apryx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_aprzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_aprxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_apryy|D\_APR|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_aprzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_aprxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_apryz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_aprzz|D\_APR|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_aprxx|tds.DF\_aprxx+tds.DiT\_apr|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_apryx|tds.DF\_apryx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_aprzx|tds.DF\_aprzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_aprxy|tds.DF\_aprxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_apryy|tds.DF\_apryy+tds.DiT\_apr|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_aprzy|tds.DF\_aprzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_aprxz|tds.DF\_aprxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_apryz|tds.DF\_apryz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_aprzz|tds.DF\_aprzz+tds.DiT\_apr|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.DF\_apsxx|D\_APS|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_apsyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_apszx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_apsxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_apsyy|D\_APS|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_apszy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_apsxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_apsyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_apszz|D\_APS|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_apsxx|tds.DF\_apsxx+tds.DiT\_aps|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_apsyx|tds.DF\_apsyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_apszx|tds.DF\_apszx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_apsxy|tds.DF\_apsxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_apsyy|tds.DF\_apsyy+tds.DiT\_aps|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_apszy|tds.DF\_apszy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_apsxz|tds.DF\_apsxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_apsyz|tds.DF\_apsyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_apszz|tds.DF\_apszz+tds.DiT\_aps|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.DF\_PTxx|D\_PT+Ds|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_PTyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_PTzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_PTxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_PTyy|D\_PT+Ds|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_PTzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_PTxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_PTyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_PTzz|D\_PT+Ds|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_PTxx|tds.DF\_PTxx+tds.DiT\_PT|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_PTyx|tds.DF\_PTyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_PTzx|tds.DF\_PTzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_PTxy|tds.DF\_PTxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_PTyy|tds.DF\_PTyy+tds.DiT\_PT|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_PTzy|tds.DF\_PTzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_PTxz|tds.DF\_PTxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_PTyz|tds.DF\_PTyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_PTzz|tds.DF\_PTzz+tds.DiT\_PT|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.DF\_thxx|D\_T+Ds|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_thyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_thzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_thxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_thyy|D\_T+Ds|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_thzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_thxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_thyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_thzz|D\_T+Ds|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_thxx|tds.DF\_thxx+tds.DiT\_th|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_thyx|tds.DF\_thyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_thzx|tds.DF\_thzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_thxy|tds.DF\_thxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_thyy|tds.DF\_thyy+tds.DiT\_th|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_thzy|tds.DF\_thzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_thxz|tds.DF\_thxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_thyz|tds.DF\_thyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_thzz|tds.DF\_thzz+tds.DiT\_th|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.DF\_atxx|D\_AT+Ds|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_atyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_atzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_atxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_atyy|D\_AT+Ds|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_atzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_atxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_atyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_atzz|D\_AT+Ds|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_atxx|tds.DF\_atxx+tds.DiT\_at|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_atyx|tds.DF\_atyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_atzx|tds.DF\_atzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_atxy|tds.DF\_atxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_atyy|tds.DF\_atyy+tds.DiT\_at|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_atzy|tds.DF\_atzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_atxz|tds.DF\_atxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_atyz|tds.DF\_atyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_atzz|tds.DF\_atzz+tds.DiT\_at|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.DF\_fgxx|D\_FG|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_fgyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_fgzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_fgxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_fgyy|D\_FG|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_fgzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_fgxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_fgyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_fgzz|D\_FG|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_fgxx|tds.DF\_fgxx+tds.DiT\_fg|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_fgyx|tds.DF\_fgyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_fgzx|tds.DF\_fgzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_fgxy|tds.DF\_fgxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_fgyy|tds.DF\_fgyy+tds.DiT\_fg|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_fgzy|tds.DF\_fgzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_fgxz|tds.DF\_fgxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_fgyz|tds.DF\_fgyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_fgzz|tds.DF\_fgzz+tds.DiT\_fg|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.DF\_fixx|D\_FI|Fluid diffusion coefficient, xx-component|Domain 1||
|tds.DF\_fiyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds.DF\_fizx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds.DF\_fixy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds.DF\_fiyy|D\_FI|Fluid diffusion coefficient, yy-component|Domain 1||
|tds.DF\_fizy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds.DF\_fixz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds.DF\_fiyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds.DF\_fizz|D\_FI|Fluid diffusion coefficient, zz-component|Domain 1||
|tds.D\_fixx|tds.DF\_fixx+tds.DiT\_fi|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds.D\_fiyx|tds.DF\_fiyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds.D\_fizx|tds.DF\_fizx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds.D\_fixy|tds.DF\_fixy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds.D\_fiyy|tds.DF\_fiyy+tds.DiT\_fi|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds.D\_fizy|tds.DF\_fizy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds.D\_fixz|tds.DF\_fixz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds.D\_fiyz|tds.DF\_fiyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds.D\_fizz|tds.DF\_fizz+tds.DiT\_fi|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds.Dav\_rp|0.5\*(tds.D\_rpxx+tds.D\_rpyy)|Average diffusion coefficient|Domain 1||
|tds.Dav\_ap|0.5\*(tds.D\_apxx+tds.D\_apyy)|Average diffusion coefficient|Domain 1||
|tds.Dav\_apr|0.5\*(tds.D\_aprxx+tds.D\_apryy)|Average diffusion coefficient|Domain 1||
|tds.Dav\_aps|0.5\*(tds.D\_apsxx+tds.D\_apsyy)|Average diffusion coefficient|Domain 1||
|tds.Dav\_PT|0.5\*(tds.D\_PTxx+tds.D\_PTyy)|Average diffusion coefficient|Domain 1||
|tds.Dav\_th|0.5\*(tds.D\_thxx+tds.D\_thyy)|Average diffusion coefficient|Domain 1||
|tds.Dav\_at|0.5\*(tds.D\_atxx+tds.D\_atyy)|Average diffusion coefficient|Domain 1||
|tds.Dav\_fg|0.5\*(tds.D\_fgxx+tds.D\_fgyy)|Average diffusion coefficient|Domain 1||
|tds.Dav\_fi|0.5\*(tds.D\_fixx+tds.D\_fiyy)|Average diffusion coefficient|Domain 1||
|tds.tflux\_rpx|tds.dflux\_rpx+tds.cflux\_rpx|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_rpy|tds.dflux\_rpy+tds.cflux\_rpy|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_rpz|tds.dflux\_rpz+tds.cflux\_rpz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_rp|sqrt(tds.dflux\_rpx^2+tds.dflux\_rpy^2+tds.dflux\_rpz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_rp|sqrt(tds.tflux\_rpx^2+tds.tflux\_rpy^2+tds.tflux\_rpz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_rpx|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_rpy|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_rpz|0|Dispersive flux, z-component|Domain 1||
|tds.tflux\_apx|tds.dflux\_apx+tds.cflux\_apx|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_apy|tds.dflux\_apy+tds.cflux\_apy|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_apz|tds.dflux\_apz+tds.cflux\_apz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_ap|sqrt(tds.dflux\_apx^2+tds.dflux\_apy^2+tds.dflux\_apz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_ap|sqrt(tds.tflux\_apx^2+tds.tflux\_apy^2+tds.tflux\_apz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_apx|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_apy|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_apz|0|Dispersive flux, z-component|Domain 1||
|tds.tflux\_aprx|tds.dflux\_aprx+tds.cflux\_aprx|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_apry|tds.dflux\_apry+tds.cflux\_apry|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_aprz|tds.dflux\_aprz+tds.cflux\_aprz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_apr|sqrt(tds.dflux\_aprx^2+tds.dflux\_apry^2+tds.dflux\_aprz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_apr|sqrt(tds.tflux\_aprx^2+tds.tflux\_apry^2+tds.tflux\_aprz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_aprx|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_apry|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_aprz|0|Dispersive flux, z-component|Domain 1||
|tds.tflux\_apsx|tds.dflux\_apsx+tds.cflux\_apsx|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_apsy|tds.dflux\_apsy+tds.cflux\_apsy|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_apsz|tds.dflux\_apsz+tds.cflux\_apsz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_aps|sqrt(tds.dflux\_apsx^2+tds.dflux\_apsy^2+tds.dflux\_apsz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_aps|sqrt(tds.tflux\_apsx^2+tds.tflux\_apsy^2+tds.tflux\_apsz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_apsx|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_apsy|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_apsz|0|Dispersive flux, z-component|Domain 1||
|tds.tflux\_PTx|tds.dflux\_PTx+tds.cflux\_PTx|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_PTy|tds.dflux\_PTy+tds.cflux\_PTy|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_PTz|tds.dflux\_PTz+tds.cflux\_PTz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_PT|sqrt(tds.dflux\_PTx^2+tds.dflux\_PTy^2+tds.dflux\_PTz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_PT|sqrt(tds.tflux\_PTx^2+tds.tflux\_PTy^2+tds.tflux\_PTz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_PTx|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_PTy|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_PTz|0|Dispersive flux, z-component|Domain 1||
|tds.tflux\_thx|tds.dflux\_thx+tds.cflux\_thx|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_thy|tds.dflux\_thy+tds.cflux\_thy|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_thz|tds.dflux\_thz+tds.cflux\_thz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_th|sqrt(tds.dflux\_thx^2+tds.dflux\_thy^2+tds.dflux\_thz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_th|sqrt(tds.tflux\_thx^2+tds.tflux\_thy^2+tds.tflux\_thz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_thx|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_thy|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_thz|0|Dispersive flux, z-component|Domain 1||
|tds.tflux\_atx|tds.dflux\_atx+tds.cflux\_atx|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_aty|tds.dflux\_aty+tds.cflux\_aty|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_atz|tds.dflux\_atz+tds.cflux\_atz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_at|sqrt(tds.dflux\_atx^2+tds.dflux\_aty^2+tds.dflux\_atz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_at|sqrt(tds.tflux\_atx^2+tds.tflux\_aty^2+tds.tflux\_atz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_atx|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_aty|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_atz|0|Dispersive flux, z-component|Domain 1||
|tds.tflux\_fgx|tds.dflux\_fgx+tds.cflux\_fgx|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_fgy|tds.dflux\_fgy+tds.cflux\_fgy|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_fgz|tds.dflux\_fgz+tds.cflux\_fgz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_fg|sqrt(tds.dflux\_fgx^2+tds.dflux\_fgy^2+tds.dflux\_fgz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_fg|sqrt(tds.tflux\_fgx^2+tds.tflux\_fgy^2+tds.tflux\_fgz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_fgx|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_fgy|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_fgz|0|Dispersive flux, z-component|Domain 1||
|tds.tflux\_fix|tds.dflux\_fix+tds.cflux\_fix|Total flux, x-component|Domain 1|+ operation|
|tds.tflux\_fiy|tds.dflux\_fiy+tds.cflux\_fiy|Total flux, y-component|Domain 1|+ operation|
|tds.tflux\_fiz|tds.dflux\_fiz+tds.cflux\_fiz|Total flux, z-component|Domain 1|+ operation|
|tds.dfluxMag\_fi|sqrt(tds.dflux\_fix^2+tds.dflux\_fiy^2+tds.dflux\_fiz^2)|Diffusive flux magnitude|Domain 1||
|tds.tfluxMag\_fi|sqrt(tds.tflux\_fix^2+tds.tflux\_fiy^2+tds.tflux\_fiz^2)|Total flux magnitude|Domain 1||
|tds.dpflux\_fix|0|Dispersive flux, x-component|Domain 1||
|tds.dpflux\_fiy|0|Dispersive flux, y-component|Domain 1||
|tds.dpflux\_fiz|0|Dispersive flux, z-component|Domain 1||
|tds.rp\_material|rp\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_rpx|-tds.D\_rpxx\*rpx-tds.D\_rpxy\*rpy|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_rpy|-tds.D\_rpyx\*rpx-tds.D\_rpyy\*rpy|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_rpz|-tds.D\_rpzx\*rpx-tds.D\_rpzy\*rpy|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_rpx|rpx|Concentration gradient, x-component|Domain 1||
|tds.grad\_rpy|rpy|Concentration gradient, y-component|Domain 1||
|tds.grad\_rpz|0|Concentration gradient, z-component|Domain 1||
|tds.ap\_material|ap\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_apx|-tds.D\_apxx\*apx-tds.D\_apxy\*apy|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_apy|-tds.D\_apyx\*apx-tds.D\_apyy\*apy|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_apz|-tds.D\_apzx\*apx-tds.D\_apzy\*apy|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_apx|apx|Concentration gradient, x-component|Domain 1||
|tds.grad\_apy|apy|Concentration gradient, y-component|Domain 1||
|tds.grad\_apz|0|Concentration gradient, z-component|Domain 1||
|tds.apr\_material|apr\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_aprx|-tds.D\_aprxx\*aprx-tds.D\_aprxy\*apry|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_apry|-tds.D\_apryx\*aprx-tds.D\_apryy\*apry|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_aprz|-tds.D\_aprzx\*aprx-tds.D\_aprzy\*apry|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_aprx|aprx|Concentration gradient, x-component|Domain 1||
|tds.grad\_apry|apry|Concentration gradient, y-component|Domain 1||
|tds.grad\_aprz|0|Concentration gradient, z-component|Domain 1||
|tds.aps\_material|aps\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_apsx|-tds.D\_apsxx\*apsx-tds.D\_apsxy\*apsy|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_apsy|-tds.D\_apsyx\*apsx-tds.D\_apsyy\*apsy|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_apsz|-tds.D\_apszx\*apsx-tds.D\_apszy\*apsy|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_apsx|apsx|Concentration gradient, x-component|Domain 1||
|tds.grad\_apsy|apsy|Concentration gradient, y-component|Domain 1||
|tds.grad\_apsz|0|Concentration gradient, z-component|Domain 1||
|tds.PT\_material|PT\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_PTx|-tds.D\_PTxx\*PTx-tds.D\_PTxy\*PTy|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_PTy|-tds.D\_PTyx\*PTx-tds.D\_PTyy\*PTy|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_PTz|-tds.D\_PTzx\*PTx-tds.D\_PTzy\*PTy|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_PTx|PTx|Concentration gradient, x-component|Domain 1||
|tds.grad\_PTy|PTy|Concentration gradient, y-component|Domain 1||
|tds.grad\_PTz|0|Concentration gradient, z-component|Domain 1||
|tds.th\_material|th\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_thx|-tds.D\_thxx\*thx-tds.D\_thxy\*thy|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_thy|-tds.D\_thyx\*thx-tds.D\_thyy\*thy|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_thz|-tds.D\_thzx\*thx-tds.D\_thzy\*thy|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_thx|thx|Concentration gradient, x-component|Domain 1||
|tds.grad\_thy|thy|Concentration gradient, y-component|Domain 1||
|tds.grad\_thz|0|Concentration gradient, z-component|Domain 1||
|tds.at\_material|at\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_atx|-tds.D\_atxx\*atx-tds.D\_atxy\*aty|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_aty|-tds.D\_atyx\*atx-tds.D\_atyy\*aty|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_atz|-tds.D\_atzx\*atx-tds.D\_atzy\*aty|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_atx|atx|Concentration gradient, x-component|Domain 1||
|tds.grad\_aty|aty|Concentration gradient, y-component|Domain 1||
|tds.grad\_atz|0|Concentration gradient, z-component|Domain 1||
|tds.fg\_material|fg\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_fgx|-tds.D\_fgxx\*fgx-tds.D\_fgxy\*fgy|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_fgy|-tds.D\_fgyx\*fgx-tds.D\_fgyy\*fgy|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_fgz|-tds.D\_fgzx\*fgx-tds.D\_fgzy\*fgy|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_fgx|fgx|Concentration gradient, x-component|Domain 1||
|tds.grad\_fgy|fgy|Concentration gradient, y-component|Domain 1||
|tds.grad\_fgz|0|Concentration gradient, z-component|Domain 1||
|tds.fi\_material|fi\*spatial.detF|Concentration|Domain 1||
|tds.dflux\_fix|-tds.D\_fixx\*fix-tds.D\_fixy\*fiy|Diffusive flux, x-component|Domain 1|+ operation|
|tds.dflux\_fiy|-tds.D\_fiyx\*fix-tds.D\_fiyy\*fiy|Diffusive flux, y-component|Domain 1|+ operation|
|tds.dflux\_fiz|-tds.D\_fizx\*fix-tds.D\_fizy\*fiy|Diffusive flux, z-component|Domain 1|+ operation|
|tds.grad\_fix|fix|Concentration gradient, x-component|Domain 1||
|tds.grad\_fiy|fiy|Concentration gradient, y-component|Domain 1||
|tds.grad\_fiz|0|Concentration gradient, z-component|Domain 1||
|tds.cflux\_rpx|rp\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_rpy|rp\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_rpz|rp\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_rp|sqrt(tds.cflux\_rpx^2+tds.cflux\_rpy^2+tds.cflux\_rpz^2)|Convective flux magnitude|Domain 1||
|tds.cflux\_apx|ap\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_apy|ap\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_apz|ap\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_ap|sqrt(tds.cflux\_apx^2+tds.cflux\_apy^2+tds.cflux\_apz^2)|Convective flux magnitude|Domain 1||
|tds.cflux\_aprx|apr\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_apry|apr\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_aprz|apr\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_apr|sqrt(tds.cflux\_aprx^2+tds.cflux\_apry^2+tds.cflux\_aprz^2)|Convective flux magnitude|Domain 1||
|tds.cflux\_apsx|aps\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_apsy|aps\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_apsz|aps\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_aps|sqrt(tds.cflux\_apsx^2+tds.cflux\_apsy^2+tds.cflux\_apsz^2)|Convective flux magnitude|Domain 1||
|tds.cflux\_PTx|PT\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_PTy|PT\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_PTz|PT\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_PT|sqrt(tds.cflux\_PTx^2+tds.cflux\_PTy^2+tds.cflux\_PTz^2)|Convective flux magnitude|Domain 1||
|tds.cflux\_thx|th\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_thy|th\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_thz|th\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_th|sqrt(tds.cflux\_thx^2+tds.cflux\_thy^2+tds.cflux\_thz^2)|Convective flux magnitude|Domain 1||
|tds.cflux\_atx|at\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_aty|at\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_atz|at\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_at|sqrt(tds.cflux\_atx^2+tds.cflux\_aty^2+tds.cflux\_atz^2)|Convective flux magnitude|Domain 1||
|tds.cflux\_fgx|fg\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_fgy|fg\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_fgz|fg\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_fg|sqrt(tds.cflux\_fgx^2+tds.cflux\_fgy^2+tds.cflux\_fgz^2)|Convective flux magnitude|Domain 1||
|tds.cflux\_fix|fi\*tds.u|Convective flux, x-component|Domain 1||
|tds.cflux\_fiy|fi\*tds.v|Convective flux, y-component|Domain 1||
|tds.cflux\_fiz|fi\*tds.w|Convective flux, z-component|Domain 1||
|tds.cfluxMag\_fi|sqrt(tds.cflux\_fix^2+tds.cflux\_fiy^2+tds.cflux\_fiz^2)|Convective flux magnitude|Domain 1||
|tds.isodiff|tds.Diso\*(-test(rpx)\*rpx-test(rpy)\*rpy-test(apx)\*apx-test(apy)\*apy-test(aprx)\*aprx-test(apry)\*apry-test(apsx)\*apsx-test(apsy)\*apsy-test(PTx)\*PTx-test(PTy)\*PTy-test(thx)\*thx-test(thy)\*thy-test(atx)\*atx-test(aty)\*aty-test(fgx)\*fgx-test(fgy)\*fgy-test(fix)\*fix-test(fiy)\*fiy)|Isotropic diffusion|Domain 1|+ operation|
|tds.bndFlux\_rp|-dflux\_spatial(rp)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.bndFlux\_ap|-dflux\_spatial(ap)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.bndFlux\_apr|-dflux\_spatial(apr)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.bndFlux\_aps|-dflux\_spatial(aps)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.bndFlux\_PT|-dflux\_spatial(PT)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.bndFlux\_th|-dflux\_spatial(th)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.bndFlux\_at|-dflux\_spatial(at)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.bndFlux\_fg|-dflux\_spatial(fg)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.bndFlux\_fi|-dflux\_spatial(fi)/tds.d|Boundary flux|Boundaries 1–8|Meta|
|tds.helem|h\_spatial|Element size|Domain 1||
|tds.glim\_mass|0.1\[mol/m^3]/tds.helem|Lower gradient limit|Domain 1||
|tds.Rlin\_rp|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_rp|rpt-(d(tds.D\_rpxx,x)+d(tds.D\_rpxy,y))\*rpx-(d(tds.D\_rpyx,x)+d(tds.D\_rpyy,y))\*rpy+tds.u\*rpx+tds.v\*rpy-rp\*tds.Rlin\_rp-tds.R\_rp|Equation residual|Domain 1||
|tds.Rlin\_ap|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_ap|apt-(d(tds.D\_apxx,x)+d(tds.D\_apxy,y))\*apx-(d(tds.D\_apyx,x)+d(tds.D\_apyy,y))\*apy+tds.u\*apx+tds.v\*apy-ap\*tds.Rlin\_ap-tds.R\_ap|Equation residual|Domain 1||
|tds.Rlin\_apr|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_apr|aprt-(d(tds.D\_aprxx,x)+d(tds.D\_aprxy,y))\*aprx-(d(tds.D\_apryx,x)+d(tds.D\_apryy,y))\*apry+tds.u\*aprx+tds.v\*apry-apr\*tds.Rlin\_apr-tds.R\_apr|Equation residual|Domain 1||
|tds.Rlin\_aps|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_aps|apst-(d(tds.D\_apsxx,x)+d(tds.D\_apsxy,y))\*apsx-(d(tds.D\_apsyx,x)+d(tds.D\_apsyy,y))\*apsy+tds.u\*apsx+tds.v\*apsy-aps\*tds.Rlin\_aps-tds.R\_aps|Equation residual|Domain 1||
|tds.Rlin\_PT|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_PT|PTt-(d(tds.D\_PTxx,x)+d(tds.D\_PTxy,y))\*PTx-(d(tds.D\_PTyx,x)+d(tds.D\_PTyy,y))\*PTy+tds.u\*PTx+tds.v\*PTy-PT\*tds.Rlin\_PT-tds.R\_PT|Equation residual|Domain 1||
|tds.Rlin\_th|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_th|tht-(d(tds.D\_thxx,x)+d(tds.D\_thxy,y))\*thx-(d(tds.D\_thyx,x)+d(tds.D\_thyy,y))\*thy+tds.u\*thx+tds.v\*thy-th\*tds.Rlin\_th-tds.R\_th|Equation residual|Domain 1||
|tds.Rlin\_at|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_at|att-(d(tds.D\_atxx,x)+d(tds.D\_atxy,y))\*atx-(d(tds.D\_atyx,x)+d(tds.D\_atyy,y))\*aty+tds.u\*atx+tds.v\*aty-at\*tds.Rlin\_at-tds.R\_at|Equation residual|Domain 1||
|tds.Rlin\_fg|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_fg|fgt-(d(tds.D\_fgxx,x)+d(tds.D\_fgxy,y))\*fgx-(d(tds.D\_fgyx,x)+d(tds.D\_fgyy,y))\*fgy+tds.u\*fgx+tds.v\*fgy-fg\*tds.Rlin\_fg-tds.R\_fg|Equation residual|Domain 1||
|tds.Rlin\_fi|0|Linear source term coefficient|Domain 1|+ operation|
|tds.Res\_fi|fit-(d(tds.D\_fixx,x)+d(tds.D\_fixy,y))\*fix-(d(tds.D\_fiyx,x)+d(tds.D\_fiyy,y))\*fiy+tds.u\*fix+tds.v\*fiy-fi\*tds.Rlin\_fi-tds.R\_fi|Equation residual|Domain 1||
|tds.Diso|h\_spatial\*tds.delid\_mass\*sqrt(tds.u^2+tds.v^2+eps)|Isotropic diffusion coefficient|Domain 1||
|tds.delid\_mass|0.25|Tuning parameter|Domain 1||

#### Shape functions

|**Name**|**Shape function**|**Description**|**Shape frame**|**Selection**|
|-|-|-|-|-|
|rp|Lagrange (Linear)|Molar concentration, rp|Spatial|Domain 1|
|ap|Lagrange (Linear)|Molar concentration, ap|Spatial|Domain 1|
|apr|Lagrange (Linear)|Molar concentration, apr|Spatial|Domain 1|
|aps|Lagrange (Linear)|Molar concentration, aps|Spatial|Domain 1|
|PT|Lagrange (Linear)|Molar concentration, PT|Spatial|Domain 1|
|th|Lagrange (Linear)|Molar concentration, th|Spatial|Domain 1|
|at|Lagrange (Linear)|Molar concentration, at|Spatial|Domain 1|
|fg|Lagrange (Linear)|Molar concentration, fg|Spatial|Domain 1|
|fi|Lagrange (Linear)|Molar concentration, fi|Spatial|Domain 1|

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|(-rpt\*test(rp)+tds.dflux\_rpx\*test(rpx)+tds.dflux\_rpy\*test(rpy))\*tds.d|2|Spatial|Domain 1|
|(-apt\*test(ap)+tds.dflux\_apx\*test(apx)+tds.dflux\_apy\*test(apy))\*tds.d|2|Spatial|Domain 1|
|(-aprt\*test(apr)+tds.dflux\_aprx\*test(aprx)+tds.dflux\_apry\*test(apry))\*tds.d|2|Spatial|Domain 1|
|(-apst\*test(aps)+tds.dflux\_apsx\*test(apsx)+tds.dflux\_apsy\*test(apsy))\*tds.d|2|Spatial|Domain 1|
|(-PTt\*test(PT)+tds.dflux\_PTx\*test(PTx)+tds.dflux\_PTy\*test(PTy))\*tds.d|2|Spatial|Domain 1|
|(-tht\*test(th)+tds.dflux\_thx\*test(thx)+tds.dflux\_thy\*test(thy))\*tds.d|2|Spatial|Domain 1|
|(-att\*test(at)+tds.dflux\_atx\*test(atx)+tds.dflux\_aty\*test(aty))\*tds.d|2|Spatial|Domain 1|
|(-fgt\*test(fg)+tds.dflux\_fgx\*test(fgx)+tds.dflux\_fgy\*test(fgy))\*tds.d|2|Spatial|Domain 1|
|(-fit\*test(fi)+tds.dflux\_fix\*test(fix)+tds.dflux\_fiy\*test(fiy))\*tds.d|2|Spatial|Domain 1|
|-(tds.u\*rpx+tds.v\*rpy)\*test(rp)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_rp\*test(rp)\*tds.d|2|Spatial|Boundaries 1–8|
|-(tds.u\*apx+tds.v\*apy)\*test(ap)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_ap\*test(ap)\*tds.d|2|Spatial|Boundaries 1–8|
|-(tds.u\*aprx+tds.v\*apry)\*test(apr)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_apr\*test(apr)\*tds.d|2|Spatial|Boundaries 1–8|
|-(tds.u\*apsx+tds.v\*apsy)\*test(aps)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_aps\*test(aps)\*tds.d|2|Spatial|Boundaries 1–8|
|-(tds.u\*PTx+tds.v\*PTy)\*test(PT)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_PT\*test(PT)\*tds.d|2|Spatial|Boundaries 1–8|
|-(tds.u\*thx+tds.v\*thy)\*test(th)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_th\*test(th)\*tds.d|2|Spatial|Boundaries 1–8|
|-(tds.u\*atx+tds.v\*aty)\*test(at)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_at\*test(at)\*tds.d|2|Spatial|Boundaries 1–8|
|-(tds.u\*fgx+tds.v\*fgy)\*test(fg)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_fg\*test(fg)\*tds.d|2|Spatial|Boundaries 1–8|
|-(tds.u\*fix+tds.v\*fiy)\*test(fi)\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.cbf\_fi\*test(fi)\*tds.d|2|Spatial|Boundaries 1–8|
|tds.streamline\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.crosswind\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|
|tds.isodiff\*(isScalingSystemDomain==0)\*tds.d|2|Spatial|Domain 1|

### No Flux 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: All boundaries|

Equations

#### Convection

Settings

|**Description**|**Value**|
|-|-|
|Include|On|

### Initial Values 1

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

#### Initial Values

Settings

|**Description**|**Value**|
|-|-|
|Concentration|{c\_RP0, c\_AP0, c\_adp0, c\_txa20, c\_pT0, c\_T0, c\_aT0, c\_Fg0, 0}|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds.c0\_rp|c\_RP0|Concentration|Domain 1|+ operation|
|tds.c0\_ap|c\_AP0|Concentration|Domain 1|+ operation|
|tds.c0\_apr|c\_adp0|Concentration|Domain 1|+ operation|
|tds.c0\_aps|c\_txa20|Concentration|Domain 1|+ operation|
|tds.c0\_PT|c\_pT0|Concentration|Domain 1|+ operation|
|tds.c0\_th|c\_T0|Concentration|Domain 1|+ operation|
|tds.c0\_at|c\_aT0|Concentration|Domain 1|+ operation|
|tds.c0\_fg|c\_Fg0|Concentration|Domain 1|+ operation|
|tds.c0\_fi|0|Concentration|Domain 1|+ operation|

### Flux 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundaries 1–4, 6–7|

Equations

#### Convection

Settings

|**Description**|**Value**|
|-|-|
|Include|On|

#### Inward Flux

Settings

|**Description**|**Value**|
|-|-|
|Flux type|General inward flux|
|Species rp|On|
|Species ap|On|
|Species apr|On|
|Species aps|On|
|Species PT|On|
|Species th|On|
|Species at|Off|
|Species fg|Off|
|Species fi|Off|
||{(-if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_rs\*RP, 0) - if(spf.sr<lss, Sat(M)\*k\_rs\*RP, 0))\*step2t(t), (-((if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_as\*AP, 0) + if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Mas/M\_inf\*k\_as\*AP, 0))) - ((if(spf.sr<lss, Sat(M)\*k\_as\*AP, 0) + if(spf.sr<lss, Mas/M\_inf\*k\_as\*AP, 0))))\*step2t(t), ((if(d(spf.sr, x)<sgt, lambda\*(L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_rs\*RP, 0) + if(spf.sr<lss, lambda\*Sat(M)\*k\_rs\*RP, 0)))\*step2t(t), Mat\*s\_t\*step2t(t), -beta\*(phi\_at\*Mat)\*PT\*step2t(t), beta\*(phi\_at\*Mat)\*PT\*step2t(t), 0, 0, 0}|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds.cbf\_rp|rp\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.cbf\_ap|ap\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.cbf\_apr|apr\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.cbf\_aps|aps\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.cbf\_PT|PT\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.cbf\_th|th\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.cbf\_at|at\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.cbf\_fg|fg\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.cbf\_fi|fi\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 1–4, 6–7||
|tds.fl1.nmflow\_rp|tds.fl1.int(tds.ntflux\_rp)\*tds.d|Normal molar flow rate|Global||
|tds.fl1.nmflow\_ap|tds.fl1.int(tds.ntflux\_ap)\*tds.d|Normal molar flow rate|Global||
|tds.fl1.nmflow\_apr|tds.fl1.int(tds.ntflux\_apr)\*tds.d|Normal molar flow rate|Global||
|tds.fl1.nmflow\_aps|tds.fl1.int(tds.ntflux\_aps)\*tds.d|Normal molar flow rate|Global||
|tds.fl1.nmflow\_PT|tds.fl1.int(tds.ntflux\_PT)\*tds.d|Normal molar flow rate|Global||
|tds.fl1.nmflow\_th|tds.fl1.int(tds.ntflux\_th)\*tds.d|Normal molar flow rate|Global||
|tds.fl1.nmflow\_at|tds.fl1.int(tds.ntflux\_at)\*tds.d|Normal molar flow rate|Global||
|tds.fl1.nmflow\_fg|tds.fl1.int(tds.ntflux\_fg)\*tds.d|Normal molar flow rate|Global||
|tds.fl1.nmflow\_fi|tds.fl1.int(tds.ntflux\_fi)\*tds.d|Normal molar flow rate|Global||

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|(-if(d(spf.sr,x)<sgt,L\*abs(d(spf.sr,x))\*Sat(M)\*k\_rs\*RP/gamma\_m,0)-if(spf.sr<lss,Sat(M)\*k\_rs\*RP,0))\*step2t(t)\*test(rp)\*tds.d|2|Spatial|Boundaries 1–4, 6–7|
|(-if(d(spf.sr,x)<sgt,L\*abs(d(spf.sr,x))\*Sat(M)\*k\_as\*AP/gamma\_m,0)-if(d(spf.sr,x)<sgt,L\*abs(d(spf.sr,x))\*Mas\*k\_as\*AP/(gamma\_m\*M\_inf),0)-if(spf.sr<lss,Sat(M)\*k\_as\*AP,0)-if(spf.sr<lss,Mas\*k\_as\*AP/M\_inf,0))\*step2t(t)\*test(ap)\*tds.d|2|Spatial|Boundaries 1–4, 6–7|
|(if(d(spf.sr,x)<sgt,lambda\*L\*abs(d(spf.sr,x))\*Sat(M)\*k\_rs\*RP/gamma\_m,0)+if(spf.sr<lss,lambda\*Sat(M)\*k\_rs\*RP,0))\*step2t(t)\*test(apr)\*tds.d|2|Spatial|Boundaries 1–4, 6–7|
|Mat\*s\_t\*step2t(t)\*test(aps)\*tds.d|2|Spatial|Boundaries 1–4, 6–7|
|-beta\*phi\_at\*Mat\*PT\*step2t(t)\*test(PT)\*tds.d|2|Spatial|Boundaries 1–4, 6–7|
|beta\*phi\_at\*Mat\*PT\*step2t(t)\*test(th)\*tds.d|2|Spatial|Boundaries 1–4, 6–7|
|0|2|Spatial|Boundaries 1–4, 6–7|
|0|2|Spatial|Boundaries 1–4, 6–7|
|0|2|Spatial|Boundaries 1–4, 6–7|

### Reactions 1

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: Domain 1|

Equations

#### Reaction Rates

Settings

|**Description**|**Value**|
|-|-|
|Total rate expression|User defined|
|Total rate expression|-k\_pa(kpa\_chem(Omega(T, APR, APS)), kpa\_mech(spf.sr))\*RP|
|Total rate expression|User defined|
|Total rate expression|k\_pa(kpa\_chem(Omega(T, APR, APS)), kpa\_mech(spf.sr))\*RP|
|Total rate expression|User defined|
|Total rate expression|lambda\*k\_pa(kpa\_chem(Omega(T, APR, APS)), kpa\_mech(spf.sr))\*RP|
|Total rate expression|User defined|
|Total rate expression|s\_t\*AP - k\_i\*APS|
|Total rate expression|User defined|
|Total rate expression|-beta\*PT\*(phi\_rt\*RP + phi\_at\*AP)|
|Total rate expression|User defined|
|Total rate expression|PT\*(phi\_at\*AP + phi\_rt\*RP)\*beta - Gamma(T, AT)\*T|
|Total rate expression|User defined|
|Total rate expression|-Gamma(AT, T)\*T|
|Total rate expression|User defined|
|Total rate expression|-(kfi\*FG\*T)/(kmfi + FG)|
|Total rate expression|User defined|
|Total rate expression|(kfi\*FG\*T)/(kmfi + FG)|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds.R\_rp|tds.reac1.R\_rp|Total rate expression|Domain 1|+ operation|
|tds.R\_ap|tds.reac1.R\_ap|Total rate expression|Domain 1|+ operation|
|tds.R\_apr|tds.reac1.R\_apr|Total rate expression|Domain 1|+ operation|
|tds.R\_aps|tds.reac1.R\_aps|Total rate expression|Domain 1|+ operation|
|tds.R\_PT|tds.reac1.R\_PT|Total rate expression|Domain 1|+ operation|
|tds.R\_th|tds.reac1.R\_th|Total rate expression|Domain 1|+ operation|
|tds.R\_at|tds.reac1.R\_at|Total rate expression|Domain 1|+ operation|
|tds.R\_fg|tds.reac1.R\_fg|Total rate expression|Domain 1|+ operation|
|tds.R\_fi|tds.reac1.R\_fi|Total rate expression|Domain 1|+ operation|
|tds.reac1.R\_rp|model.input.R\_rp|Total rate expression|Domain 1|Meta|
|tds.reac1.R\_ap|model.input.R\_ap|Total rate expression|Domain 1|Meta|
|tds.reac1.R\_apr|model.input.R\_apr|Total rate expression|Domain 1|Meta|
|tds.reac1.R\_aps|model.input.R\_aps|Total rate expression|Domain 1|Meta|
|tds.reac1.R\_PT|model.input.R\_PT|Total rate expression|Domain 1|Meta|
|tds.reac1.R\_th|model.input.R\_th|Total rate expression|Domain 1|Meta|
|tds.reac1.R\_at|model.input.R\_at|Total rate expression|Domain 1|Meta|
|tds.reac1.R\_fg|model.input.R\_fg|Total rate expression|Domain 1|Meta|
|tds.reac1.R\_fi|model.input.R\_fi|Total rate expression|Domain 1|Meta|

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|tds.reac1.R\_rp\*test(rp)\*tds.d|2|Spatial|Domain 1|
|tds.reac1.R\_ap\*test(ap)\*tds.d|2|Spatial|Domain 1|
|tds.reac1.R\_apr\*test(apr)\*tds.d|2|Spatial|Domain 1|
|tds.reac1.R\_aps\*test(aps)\*tds.d|2|Spatial|Domain 1|
|tds.reac1.R\_PT\*test(PT)\*tds.d|2|Spatial|Domain 1|
|tds.reac1.R\_th\*test(th)\*tds.d|2|Spatial|Domain 1|
|tds.reac1.R\_at\*test(at)\*tds.d|2|Spatial|Domain 1|
|tds.reac1.R\_fg\*test(fg)\*tds.d|2|Spatial|Domain 1|
|tds.reac1.R\_fi\*test(fi)\*tds.d|2|Spatial|Domain 1|

### Flusso uscente 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundary 8|

Equations

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds.out1.c0\_avg\_rp|tds.out1.int(rp\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.c0\_avg\_ap|tds.out1.int(ap\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.c0\_avg\_apr|tds.out1.int(apr\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.c0\_avg\_aps|tds.out1.int(aps\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.c0\_avg\_PT|tds.out1.int(PT\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.c0\_avg\_th|tds.out1.int(th\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.c0\_avg\_at|tds.out1.int(at\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.c0\_avg\_fg|tds.out1.int(fg\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.c0\_avg\_fi|tds.out1.int(fi\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.out1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.out1.nmflow\_rp|tds.out1.int(tds.ntflux\_rp)\*tds.d|Normal molar flow rate|Global||
|tds.out1.nmflow\_ap|tds.out1.int(tds.ntflux\_ap)\*tds.d|Normal molar flow rate|Global||
|tds.out1.nmflow\_apr|tds.out1.int(tds.ntflux\_apr)\*tds.d|Normal molar flow rate|Global||
|tds.out1.nmflow\_aps|tds.out1.int(tds.ntflux\_aps)\*tds.d|Normal molar flow rate|Global||
|tds.out1.nmflow\_PT|tds.out1.int(tds.ntflux\_PT)\*tds.d|Normal molar flow rate|Global||
|tds.out1.nmflow\_th|tds.out1.int(tds.ntflux\_th)\*tds.d|Normal molar flow rate|Global||
|tds.out1.nmflow\_at|tds.out1.int(tds.ntflux\_at)\*tds.d|Normal molar flow rate|Global||
|tds.out1.nmflow\_fg|tds.out1.int(tds.ntflux\_fg)\*tds.d|Normal molar flow rate|Global||
|tds.out1.nmflow\_fi|tds.out1.int(tds.ntflux\_fi)\*tds.d|Normal molar flow rate|Global||

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|0|2|Spatial|Boundary 8|
|0|2|Spatial|Boundary 8|
|0|2|Spatial|Boundary 8|
|0|2|Spatial|Boundary 8|
|0|2|Spatial|Boundary 8|
|0|2|Spatial|Boundary 8|
|0|2|Spatial|Boundary 8|
|0|2|Spatial|Boundary 8|
|0|2|Spatial|Boundary 8|

### Flusso entrante 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundary 5|

Equations

#### Concentration

Settings

|**Description**|**Value**|
|-|-|
|Concentration|{c\_RP0, c\_AP0, 0, 0, c\_pT0, 0, c\_aT0, c\_Fg0, 0}|

#### Boundary Condition Type

Settings

|**Description**|**Value**|
|-|-|
|Boundary condition type|Flux (Danckwerts)|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds.cbf\_rp|rp\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.cbf\_ap|ap\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.cbf\_apr|apr\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.cbf\_aps|aps\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.cbf\_PT|PT\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.cbf\_th|th\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.cbf\_at|at\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.cbf\_fg|fg\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.cbf\_fi|fi\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundary 5||
|tds.c0\_rp|c\_RP0|Concentration|Boundary 5|+ operation|
|tds.c0\_ap|c\_AP0|Concentration|Boundary 5|+ operation|
|tds.c0\_apr|0|Concentration|Boundary 5|+ operation|
|tds.c0\_aps|0|Concentration|Boundary 5|+ operation|
|tds.c0\_PT|c\_pT0|Concentration|Boundary 5|+ operation|
|tds.c0\_th|0|Concentration|Boundary 5|+ operation|
|tds.c0\_at|c\_aT0|Concentration|Boundary 5|+ operation|
|tds.c0\_fg|c\_Fg0|Concentration|Boundary 5|+ operation|
|tds.c0\_fi|0|Concentration|Boundary 5|+ operation|
|tds.nU|tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz|Velocity field|Boundary 5||
|tds.in1.c0\_avg\_rp|tds.in1.int(rp\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.c0\_avg\_ap|tds.in1.int(ap\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.c0\_avg\_apr|tds.in1.int(apr\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.c0\_avg\_aps|tds.in1.int(aps\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.c0\_avg\_PT|tds.in1.int(PT\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.c0\_avg\_th|tds.in1.int(th\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.c0\_avg\_at|tds.in1.int(at\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.c0\_avg\_fg|tds.in1.int(fg\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.c0\_avg\_fi|tds.in1.int(fi\*(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz))/tds.in1.int(tds.u\*tds.nx+tds.v\*tds.ny+tds.w\*tds.nz)|Average concentration|Global||
|tds.in1.nmflow\_rp|tds.in1.int(tds.ntflux\_rp)\*tds.d|Normal molar flow rate|Global||
|tds.in1.nmflow\_ap|tds.in1.int(tds.ntflux\_ap)\*tds.d|Normal molar flow rate|Global||
|tds.in1.nmflow\_apr|tds.in1.int(tds.ntflux\_apr)\*tds.d|Normal molar flow rate|Global||
|tds.in1.nmflow\_aps|tds.in1.int(tds.ntflux\_aps)\*tds.d|Normal molar flow rate|Global||
|tds.in1.nmflow\_PT|tds.in1.int(tds.ntflux\_PT)\*tds.d|Normal molar flow rate|Global||
|tds.in1.nmflow\_th|tds.in1.int(tds.ntflux\_th)\*tds.d|Normal molar flow rate|Global||
|tds.in1.nmflow\_at|tds.in1.int(tds.ntflux\_at)\*tds.d|Normal molar flow rate|Global||
|tds.in1.nmflow\_fg|tds.in1.int(tds.ntflux\_fg)\*tds.d|Normal molar flow rate|Global||
|tds.in1.nmflow\_fi|tds.in1.int(tds.ntflux\_fi)\*tds.d|Normal molar flow rate|Global||

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|-tds.c0\_rp\*tds.nU\*test(rp)\*tds.d|2|Spatial|Boundary 5|
|-tds.c0\_ap\*tds.nU\*test(ap)\*tds.d|2|Spatial|Boundary 5|
|-tds.c0\_apr\*tds.nU\*test(apr)\*tds.d|2|Spatial|Boundary 5|
|-tds.c0\_aps\*tds.nU\*test(aps)\*tds.d|2|Spatial|Boundary 5|
|-tds.c0\_PT\*tds.nU\*test(PT)\*tds.d|2|Spatial|Boundary 5|
|-tds.c0\_th\*tds.nU\*test(th)\*tds.d|2|Spatial|Boundary 5|
|-tds.c0\_at\*tds.nU\*test(at)\*tds.d|2|Spatial|Boundary 5|
|-tds.c0\_fg\*tds.nU\*test(fg)\*tds.d|2|Spatial|Boundary 5|
|-tds.c0\_fi\*tds.nU\*test(fi)\*tds.d|2|Spatial|Boundary 5|

### Flusso 2

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundaries 3–4|

Equations

#### Convection

Settings

|**Description**|**Value**|
|-|-|
|Include|On|

#### Inward Flux

Settings

|**Description**|**Value**|
|-|-|
|Flux type|General inward flux|
|Species rp|On|
|Species ap|On|
|Species apr|On|
|Species aps|On|
|Species PT|On|
|Species th|On|
|Species at|Off|
|Species fg|Off|
|Species fi|Off|
||{-(Sat(M)\*k\_rs\*RP)\*step2t(t), -((Sat(M)\*k\_as\*AP) + (Mas/M\_inf\*k\_as\*AP))\*step2t(t), (lambda\*Sat(M)\*k\_rs\*RP)\*step2t(t), Mat\*s\_t\*step2t(t), -beta\*(phi\_at\*Mat)\*PT\*step2t(t), beta\*(phi\_at\*Mat)\*PT\*step2t(t), 0, 0, 0}|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds.cbf\_rp|rp\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.cbf\_ap|ap\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.cbf\_apr|apr\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.cbf\_aps|aps\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.cbf\_PT|PT\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.cbf\_th|th\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.cbf\_at|at\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.cbf\_fg|fg\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.cbf\_fi|fi\*(tds.u\*tds.nxmesh+tds.v\*tds.nymesh+tds.w\*tds.nzmesh)|Convective boundary flux|Boundaries 3–4||
|tds.fl2.nmflow\_rp|tds.fl2.int(tds.ntflux\_rp)\*tds.d|Normal molar flow rate|Global||
|tds.fl2.nmflow\_ap|tds.fl2.int(tds.ntflux\_ap)\*tds.d|Normal molar flow rate|Global||
|tds.fl2.nmflow\_apr|tds.fl2.int(tds.ntflux\_apr)\*tds.d|Normal molar flow rate|Global||
|tds.fl2.nmflow\_aps|tds.fl2.int(tds.ntflux\_aps)\*tds.d|Normal molar flow rate|Global||
|tds.fl2.nmflow\_PT|tds.fl2.int(tds.ntflux\_PT)\*tds.d|Normal molar flow rate|Global||
|tds.fl2.nmflow\_th|tds.fl2.int(tds.ntflux\_th)\*tds.d|Normal molar flow rate|Global||
|tds.fl2.nmflow\_at|tds.fl2.int(tds.ntflux\_at)\*tds.d|Normal molar flow rate|Global||
|tds.fl2.nmflow\_fg|tds.fl2.int(tds.ntflux\_fg)\*tds.d|Normal molar flow rate|Global||
|tds.fl2.nmflow\_fi|tds.fl2.int(tds.ntflux\_fi)\*tds.d|Normal molar flow rate|Global||

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|-Sat(M)\*k\_rs\*RP\*step2t(t)\*test(rp)\*tds.d|2|Spatial|Boundaries 3–4|
|-k\_as\*AP\*(Sat(M)+Mas/M\_inf)\*step2t(t)\*test(ap)\*tds.d|2|Spatial|Boundaries 3–4|
|lambda\*Sat(M)\*k\_rs\*RP\*step2t(t)\*test(apr)\*tds.d|2|Spatial|Boundaries 3–4|
|Mat\*s\_t\*step2t(t)\*test(aps)\*tds.d|2|Spatial|Boundaries 3–4|
|-beta\*phi\_at\*Mat\*PT\*step2t(t)\*test(PT)\*tds.d|2|Spatial|Boundaries 3–4|
|beta\*phi\_at\*Mat\*PT\*step2t(t)\*test(th)\*tds.d|2|Spatial|Boundaries 3–4|
|0|2|Spatial|Boundaries 3–4|
|0|2|Spatial|Boundaries 3–4|
|0|2|Spatial|Boundaries 3–4|

## Transport of Diluted Species 2

Used products

||
|-|
|COMSOL Multiphysics|
|CFD Module|

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

Equations

### Interface Settings

#### Discretization

Settings

|**Description**|**Value**|
|-|-|
|Concentration|Quadratic|

Settings

|**Description**|**Value**|
|-|-|
|Equation form|Study controlled|

#### Out-of-Plane Thickness

Settings

|**Description**|**Value**|
|-|-|
|Out-of-plane thickness|1\[m]|

#### Species Activity

Settings

|**Description**|**Value**|
|-|-|
|Species activity|Ideal|

#### Transport Mechanisms

Settings

|**Description**|**Value**|
|-|-|
|Convection|Off|
|Mass transfer in porous media|Off|

### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds2.cVar\_M|M|Species|Boundaries 1–8||
|tds2.cVar\_M|M|Species|Points 1–8||
|tds2.cVar\_Mas|Mas|Species|Boundaries 1–8||
|tds2.cVar\_Mas|Mas|Species|Points 1–8||
|tds2.cVar\_Mat|Mat|Species|Boundaries 1–8||
|tds2.cVar\_Mat|Mat|Species|Points 1–8||
|tds2.dz|1\[m]|Out-of-plane thickness|Global||
|tds2.d|tds2.dz|Out-of-plane geometry extension|Global||
|tds2.f\_M|1|Activity coefficient|Domain 1||
|tds2.f\_Mas|1|Activity coefficient|Domain 1||
|tds2.f\_Mat|1|Activity coefficient|Domain 1||
|tds2.nx|dnx|Normal vector, x-component|Boundaries 1–8||
|tds2.ny|dny|Normal vector, y-component|Boundaries 1–8||
|tds2.nz|0|Normal vector, z-component|Boundaries 1–8||
|tds2.nX|dnX|Normal vector, X-component|Boundaries 1–8||
|tds2.nY|dnY|Normal vector, Y-component|Boundaries 1–8||
|tds2.nZ|0|Normal vector, Z-component|Boundaries 1–8||
|tds2.nXg|dnXg|Normal vector, Xg-component|Boundaries 1–8||
|tds2.nYg|dnYg|Normal vector, Yg-component|Boundaries 1–8||
|tds2.nZg|0|Normal vector, Zg-component|Boundaries 1–8||
|tds2.nxmesh|dnxmesh|Normal vector (mesh), x-component|Boundaries 1–8||
|tds2.nymesh|dnymesh|Normal vector (mesh), y-component|Boundaries 1–8||
|tds2.nzmesh|0|Normal vector (mesh), z-component|Boundaries 1–8||
|tds2.nxc|nxc/tds2.ncLen|Normal vector, x-component|Boundaries 1–8||
|tds2.nyc|nyc/tds2.ncLen|Normal vector, y-component|Boundaries 1–8||
|tds2.nzc|0|Normal vector, z-component|Boundaries 1–8||
|tds2.ncLen|sqrt(nxc^2+nyc^2+eps)|Help variable|Boundaries 1–8||
|tds2.R\_M|0|Total rate expression|Domain 1|+ operation|
|tds2.cP\_M|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds2.cP\_M|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds2.KP\_M|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds2.KP\_M|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds2.Rads\_M|0|Total adsorption rate|Domain 1|+ operation|
|tds2.DiT\_M|0|Turbulent diffusivity|Domain 1||
|tds2.R\_Mas|0|Total rate expression|Domain 1|+ operation|
|tds2.cP\_Mas|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds2.cP\_Mas|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds2.KP\_Mas|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds2.KP\_Mas|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds2.Rads\_Mas|0|Total adsorption rate|Domain 1|+ operation|
|tds2.DiT\_Mas|0|Turbulent diffusivity|Domain 1||
|tds2.R\_Mat|0|Total rate expression|Domain 1|+ operation|
|tds2.cP\_Mat|0|Concentration species adsorbed to the solid|Domain 1|+ operation|
|tds2.cP\_Mat|0|Concentration species adsorbed to the solid|Boundaries 1–8|+ operation|
|tds2.KP\_Mat|0|Adsorption isotherm, first concentration derivative|Domain 1|+ operation|
|tds2.KP\_Mat|0|Adsorption isotherm, first concentration derivative|Boundaries 1–8|+ operation|
|tds2.Rads\_Mat|0|Total adsorption rate|Domain 1|+ operation|
|tds2.DiT\_Mat|0|Turbulent diffusivity|Domain 1||
|tds2.poro|1|Porosity|Domain 1||
|tds2.theta\_g|0|Gas volume fraction|Domain 1||
|tds2.theta\_l|1|Liquid volume fraction|Domain 1||
|tds2.theta|tds2.poro|Mobile fluid volume fraction|Domain 1||

### Transport Properties 1

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

Equations

#### Diffusion

Settings

|**Description**|**Value**|
|-|-|
|Source|Material|
|Material|Material 1 (mat1) {mat1}|
|Diffusion coefficient|User defined|
|Diffusion coefficient|0|
|Diffusion coefficient|User defined|
|Diffusion coefficient|0|
|Diffusion coefficient|User defined|
|Diffusion coefficient|0|

#### Coordinate System Selection

Settings

|**Description**|**Value**|
|-|-|
|Coordinate system|Global coordinate system|

#### Model Input

Settings

|**Description**|**Value**|
|-|-|
|Temperature|Common model input|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|domflux.Mx|tds2.dflux\_Mx\*tds2.d|Domain flux, x-component|Domain 1||
|domflux.My|tds2.dflux\_My\*tds2.d|Domain flux, y-component|Domain 1||
|domflux.Masx|tds2.dflux\_Masx\*tds2.d|Domain flux, x-component|Domain 1||
|domflux.Masy|tds2.dflux\_Masy\*tds2.d|Domain flux, y-component|Domain 1||
|domflux.Matx|tds2.dflux\_Matx\*tds2.d|Domain flux, x-component|Domain 1||
|domflux.Maty|tds2.dflux\_Maty\*tds2.d|Domain flux, y-component|Domain 1||
|tds2.ndflux\_M|tds2.bndFlux\_M|Normal diffusive flux|Boundaries 1–8||
|tds2.ntflux\_M|tds2.bndFlux\_M|Normal total flux|Boundaries 1–8||
|tds2.ndflux\_Mas|tds2.bndFlux\_Mas|Normal diffusive flux|Boundaries 1–8||
|tds2.ntflux\_Mas|tds2.bndFlux\_Mas|Normal total flux|Boundaries 1–8||
|tds2.ndflux\_Mat|tds2.bndFlux\_Mat|Normal diffusive flux|Boundaries 1–8||
|tds2.ntflux\_Mat|tds2.bndFlux\_Mat|Normal total flux|Boundaries 1–8||
|tds2.DF\_Mxx|0|Fluid diffusion coefficient, xx-component|Domain 1||
|tds2.DF\_Myx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds2.DF\_Mzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds2.DF\_Mxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds2.DF\_Myy|0|Fluid diffusion coefficient, yy-component|Domain 1||
|tds2.DF\_Mzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds2.DF\_Mxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds2.DF\_Myz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds2.DF\_Mzz|0|Fluid diffusion coefficient, zz-component|Domain 1||
|tds2.D\_Mxx|tds2.DF\_Mxx+tds2.DiT\_M|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds2.D\_Myx|tds2.DF\_Myx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds2.D\_Mzx|tds2.DF\_Mzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds2.D\_Mxy|tds2.DF\_Mxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds2.D\_Myy|tds2.DF\_Myy+tds2.DiT\_M|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds2.D\_Mzy|tds2.DF\_Mzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds2.D\_Mxz|tds2.DF\_Mxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds2.D\_Myz|tds2.DF\_Myz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds2.D\_Mzz|tds2.DF\_Mzz+tds2.DiT\_M|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds2.DF\_Masxx|0|Fluid diffusion coefficient, xx-component|Domain 1||
|tds2.DF\_Masyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds2.DF\_Maszx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds2.DF\_Masxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds2.DF\_Masyy|0|Fluid diffusion coefficient, yy-component|Domain 1||
|tds2.DF\_Maszy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds2.DF\_Masxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds2.DF\_Masyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds2.DF\_Maszz|0|Fluid diffusion coefficient, zz-component|Domain 1||
|tds2.D\_Masxx|tds2.DF\_Masxx+tds2.DiT\_Mas|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds2.D\_Masyx|tds2.DF\_Masyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds2.D\_Maszx|tds2.DF\_Maszx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds2.D\_Masxy|tds2.DF\_Masxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds2.D\_Masyy|tds2.DF\_Masyy+tds2.DiT\_Mas|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds2.D\_Maszy|tds2.DF\_Maszy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds2.D\_Masxz|tds2.DF\_Masxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds2.D\_Masyz|tds2.DF\_Masyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds2.D\_Maszz|tds2.DF\_Maszz+tds2.DiT\_Mas|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds2.DF\_Matxx|0|Fluid diffusion coefficient, xx-component|Domain 1||
|tds2.DF\_Matyx|0|Fluid diffusion coefficient, yx-component|Domain 1||
|tds2.DF\_Matzx|0|Fluid diffusion coefficient, zx-component|Domain 1||
|tds2.DF\_Matxy|0|Fluid diffusion coefficient, xy-component|Domain 1||
|tds2.DF\_Matyy|0|Fluid diffusion coefficient, yy-component|Domain 1||
|tds2.DF\_Matzy|0|Fluid diffusion coefficient, zy-component|Domain 1||
|tds2.DF\_Matxz|0|Fluid diffusion coefficient, xz-component|Domain 1||
|tds2.DF\_Matyz|0|Fluid diffusion coefficient, yz-component|Domain 1||
|tds2.DF\_Matzz|0|Fluid diffusion coefficient, zz-component|Domain 1||
|tds2.D\_Matxx|tds2.DF\_Matxx+tds2.DiT\_Mat|Diffusion coefficient, xx-component|Domain 1|+ operation|
|tds2.D\_Matyx|tds2.DF\_Matyx|Diffusion coefficient, yx-component|Domain 1|+ operation|
|tds2.D\_Matzx|tds2.DF\_Matzx|Diffusion coefficient, zx-component|Domain 1|+ operation|
|tds2.D\_Matxy|tds2.DF\_Matxy|Diffusion coefficient, xy-component|Domain 1|+ operation|
|tds2.D\_Matyy|tds2.DF\_Matyy+tds2.DiT\_Mat|Diffusion coefficient, yy-component|Domain 1|+ operation|
|tds2.D\_Matzy|tds2.DF\_Matzy|Diffusion coefficient, zy-component|Domain 1|+ operation|
|tds2.D\_Matxz|tds2.DF\_Matxz|Diffusion coefficient, xz-component|Domain 1|+ operation|
|tds2.D\_Matyz|tds2.DF\_Matyz|Diffusion coefficient, yz-component|Domain 1|+ operation|
|tds2.D\_Matzz|tds2.DF\_Matzz+tds2.DiT\_Mat|Diffusion coefficient, zz-component|Domain 1|+ operation|
|tds2.Dav\_M|0.5\*(tds2.D\_Mxx+tds2.D\_Myy)|Average diffusion coefficient|Domain 1||
|tds2.Dav\_Mas|0.5\*(tds2.D\_Masxx+tds2.D\_Masyy)|Average diffusion coefficient|Domain 1||
|tds2.Dav\_Mat|0.5\*(tds2.D\_Matxx+tds2.D\_Matyy)|Average diffusion coefficient|Domain 1||
|tds2.tflux\_Mx|tds2.dflux\_Mx|Total flux, x-component|Domain 1|+ operation|
|tds2.tflux\_My|tds2.dflux\_My|Total flux, y-component|Domain 1|+ operation|
|tds2.tflux\_Mz|tds2.dflux\_Mz|Total flux, z-component|Domain 1|+ operation|
|tds2.dfluxMag\_M|sqrt(tds2.dflux\_Mx^2+tds2.dflux\_My^2+tds2.dflux\_Mz^2)|Diffusive flux magnitude|Domain 1||
|tds2.tfluxMag\_M|sqrt(tds2.tflux\_Mx^2+tds2.tflux\_My^2+tds2.tflux\_Mz^2)|Total flux magnitude|Domain 1||
|tds2.dpflux\_Mx|0|Dispersive flux, x-component|Domain 1||
|tds2.dpflux\_My|0|Dispersive flux, y-component|Domain 1||
|tds2.dpflux\_Mz|0|Dispersive flux, z-component|Domain 1||
|tds2.tflux\_Masx|tds2.dflux\_Masx|Total flux, x-component|Domain 1|+ operation|
|tds2.tflux\_Masy|tds2.dflux\_Masy|Total flux, y-component|Domain 1|+ operation|
|tds2.tflux\_Masz|tds2.dflux\_Masz|Total flux, z-component|Domain 1|+ operation|
|tds2.dfluxMag\_Mas|sqrt(tds2.dflux\_Masx^2+tds2.dflux\_Masy^2+tds2.dflux\_Masz^2)|Diffusive flux magnitude|Domain 1||
|tds2.tfluxMag\_Mas|sqrt(tds2.tflux\_Masx^2+tds2.tflux\_Masy^2+tds2.tflux\_Masz^2)|Total flux magnitude|Domain 1||
|tds2.dpflux\_Masx|0|Dispersive flux, x-component|Domain 1||
|tds2.dpflux\_Masy|0|Dispersive flux, y-component|Domain 1||
|tds2.dpflux\_Masz|0|Dispersive flux, z-component|Domain 1||
|tds2.tflux\_Matx|tds2.dflux\_Matx|Total flux, x-component|Domain 1|+ operation|
|tds2.tflux\_Maty|tds2.dflux\_Maty|Total flux, y-component|Domain 1|+ operation|
|tds2.tflux\_Matz|tds2.dflux\_Matz|Total flux, z-component|Domain 1|+ operation|
|tds2.dfluxMag\_Mat|sqrt(tds2.dflux\_Matx^2+tds2.dflux\_Maty^2+tds2.dflux\_Matz^2)|Diffusive flux magnitude|Domain 1||
|tds2.tfluxMag\_Mat|sqrt(tds2.tflux\_Matx^2+tds2.tflux\_Maty^2+tds2.tflux\_Matz^2)|Total flux magnitude|Domain 1||
|tds2.dpflux\_Matx|0|Dispersive flux, x-component|Domain 1||
|tds2.dpflux\_Maty|0|Dispersive flux, y-component|Domain 1||
|tds2.dpflux\_Matz|0|Dispersive flux, z-component|Domain 1||
|tds2.M\_material|M\*spatial.detF|Concentration|Domain 1||
|tds2.dflux\_Mx|-tds2.D\_Mxx\*Mx-tds2.D\_Mxy\*My|Diffusive flux, x-component|Domain 1|+ operation|
|tds2.dflux\_My|-tds2.D\_Myx\*Mx-tds2.D\_Myy\*My|Diffusive flux, y-component|Domain 1|+ operation|
|tds2.dflux\_Mz|-tds2.D\_Mzx\*Mx-tds2.D\_Mzy\*My|Diffusive flux, z-component|Domain 1|+ operation|
|tds2.grad\_Mx|Mx|Concentration gradient, x-component|Domain 1||
|tds2.grad\_My|My|Concentration gradient, y-component|Domain 1||
|tds2.grad\_Mz|0|Concentration gradient, z-component|Domain 1||
|tds2.Mas\_material|Mas\*spatial.detF|Concentration|Domain 1||
|tds2.dflux\_Masx|-tds2.D\_Masxx\*Masx-tds2.D\_Masxy\*Masy|Diffusive flux, x-component|Domain 1|+ operation|
|tds2.dflux\_Masy|-tds2.D\_Masyx\*Masx-tds2.D\_Masyy\*Masy|Diffusive flux, y-component|Domain 1|+ operation|
|tds2.dflux\_Masz|-tds2.D\_Maszx\*Masx-tds2.D\_Maszy\*Masy|Diffusive flux, z-component|Domain 1|+ operation|
|tds2.grad\_Masx|Masx|Concentration gradient, x-component|Domain 1||
|tds2.grad\_Masy|Masy|Concentration gradient, y-component|Domain 1||
|tds2.grad\_Masz|0|Concentration gradient, z-component|Domain 1||
|tds2.Mat\_material|Mat\*spatial.detF|Concentration|Domain 1||
|tds2.dflux\_Matx|-tds2.D\_Matxx\*Matx-tds2.D\_Matxy\*Maty|Diffusive flux, x-component|Domain 1|+ operation|
|tds2.dflux\_Maty|-tds2.D\_Matyx\*Matx-tds2.D\_Matyy\*Maty|Diffusive flux, y-component|Domain 1|+ operation|
|tds2.dflux\_Matz|-tds2.D\_Matzx\*Matx-tds2.D\_Matzy\*Maty|Diffusive flux, z-component|Domain 1|+ operation|
|tds2.grad\_Matx|Matx|Concentration gradient, x-component|Domain 1||
|tds2.grad\_Maty|Maty|Concentration gradient, y-component|Domain 1||
|tds2.grad\_Matz|0|Concentration gradient, z-component|Domain 1||
|tds2.bndFlux\_M|-dflux\_spatial(M)/tds2.d|Boundary flux|Boundaries 1–8|Meta|
|tds2.bndFlux\_Mas|-dflux\_spatial(Mas)/tds2.d|Boundary flux|Boundaries 1–8|Meta|
|tds2.bndFlux\_Mat|-dflux\_spatial(Mat)/tds2.d|Boundary flux|Boundaries 1–8|Meta|
|tds2.Rlin\_M|0|Linear source term coefficient|Domain 1|+ operation|
|tds2.Res\_M|Mt-tds2.D\_Mxx\*Mxx-tds2.D\_Mxy\*Mxy-tds2.D\_Myx\*Myx-tds2.D\_Myy\*Myy-M\*tds2.Rlin\_M-tds2.R\_M|Equation residual|Domain 1||
|tds2.Rlin\_Mas|0|Linear source term coefficient|Domain 1|+ operation|
|tds2.Res\_Mas|Mast-tds2.D\_Masxx\*Masxx-tds2.D\_Masxy\*Masxy-tds2.D\_Masyx\*Masyx-tds2.D\_Masyy\*Masyy-Mas\*tds2.Rlin\_Mas-tds2.R\_Mas|Equation residual|Domain 1||
|tds2.Rlin\_Mat|0|Linear source term coefficient|Domain 1|+ operation|
|tds2.Res\_Mat|Matt-tds2.D\_Matxx\*Matxx-tds2.D\_Matxy\*Matxy-tds2.D\_Matyx\*Matyx-tds2.D\_Matyy\*Matyy-Mat\*tds2.Rlin\_Mat-tds2.R\_Mat|Equation residual|Domain 1||

#### Shape functions

|**Name**|**Shape function**|**Description**|**Shape frame**|**Selection**|
|-|-|-|-|-|
|M|Lagrange (Quadratic)|Molar concentration, M|Spatial|Domain 1|
|Mas|Lagrange (Quadratic)|Molar concentration, Mas|Spatial|Domain 1|
|Mat|Lagrange (Quadratic)|Molar concentration, Mat|Spatial|Domain 1|

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|(-Mt\*test(M)+tds2.dflux\_Mx\*test(Mx)+tds2.dflux\_My\*test(My))\*tds2.d|4|Spatial|Domain 1|
|(-Mast\*test(Mas)+tds2.dflux\_Masx\*test(Masx)+tds2.dflux\_Masy\*test(Masy))\*tds2.d|4|Spatial|Domain 1|
|(-Matt\*test(Mat)+tds2.dflux\_Matx\*test(Matx)+tds2.dflux\_Maty\*test(Maty))\*tds2.d|4|Spatial|Domain 1|
|tds2.streamline\*(isScalingSystemDomain==0)\*tds2.d|4|Spatial|Domain 1|
|tds2.crosswind\*(isScalingSystemDomain==0)\*tds2.d|6|Spatial|Domain 1|

### No Flux 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: All boundaries|

Equations

### Initial Values 1

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

#### Initial Values

Settings

|**Description**|**Value**|
|-|-|
|Concentration|{0, 0, 0}|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds2.c0\_M|0|Concentration|Domain 1|+ operation|
|tds2.c0\_Mas|0|Concentration|Domain 1|+ operation|
|tds2.c0\_Mat|0|Concentration|Domain 1|+ operation|

### Surface Reactions 1

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundaries 1–4, 6–7|

Equations

#### Surface Reaction Rate

Settings

|**Description**|**Value**|
|-|-|
|Surface reaction rate|User defined|
|Surface reaction rate|Da\*((if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_rs\*RP, 0) + if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_as\*AP, 0) + if(spf.sr<lss, Sat(M)\*(k\_rs\*RP + k\_as\*AP), 0)))\*step2t(t)|
|Surface reaction rate|User defined|
|Surface reaction rate|Da\*((if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_rs\*RP, 0) + if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_as\*AP, 0) + if(spf.sr<lss, Sat(M)\*(k\_rs\*RP + k\_as\*AP), 0)))\*step2t(t)|
|Surface reaction rate|User defined|
|Surface reaction rate|Da\*((if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_rs\*RP, 0) + if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Sat(M)\*k\_as\*AP, 0) + if(d(spf.sr, x)<sgt, (L/gamma\_m)\*abs(d(spf.sr, x))\*Mas/M\_inf\*k\_aa\*AP, 0) + if(spf.sr<lss, Sat(M)\*k\_rs\*RP + Sat(M)\*k\_as\*AP + (Mas/M\_inf)\*k\_aa\*AP, 0)))\*step2t(t)|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds2.J0\_M|model.input.J0\_M|Surface reaction rate|Boundaries 1–4, 6–7|Meta, + operation|
|tds2.J0\_Mas|model.input.J0\_Mas|Surface reaction rate|Boundaries 1–4, 6–7|Meta, + operation|
|tds2.J0\_Mat|model.input.J0\_Mat|Surface reaction rate|Boundaries 1–4, 6–7|Meta, + operation|

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|tds2.J0\_M\*test(M)\*tds2.d|4|Spatial|Boundaries 1–4, 6–7|
|tds2.J0\_Mas\*test(Mas)\*tds2.d|4|Spatial|Boundaries 1–4, 6–7|
|tds2.J0\_Mat\*test(Mat)\*tds2.d|4|Spatial|Boundaries 1–4, 6–7|

### Reazioni sulle superfici 2

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundaries 3–4|

Equations

#### Surface Reaction Rate

Settings

|**Description**|**Value**|
|-|-|
|Surface reaction rate|User defined|
|Surface reaction rate|Da\*(Sat(M)\*(k\_rs\*RP + k\_as\*AP))\*step2t(t)|
|Surface reaction rate|User defined|
|Surface reaction rate|Da\*(Sat(M)\*(k\_rs\*RP + k\_as\*AP))\*step2t(t)|
|Surface reaction rate|User defined|
|Surface reaction rate|Da\*(Sat(M)\*k\_rs\*RP + Sat(M)\*k\_as\*AP + (Mas/M\_inf)\*k\_aa\*AP)\*step2t(t)|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds2.J0\_M|model.input.J0\_M|Surface reaction rate|Boundaries 3–4|Meta, + operation|
|tds2.J0\_Mas|model.input.J0\_Mas|Surface reaction rate|Boundaries 3–4|Meta, + operation|
|tds2.J0\_Mat|model.input.J0\_Mat|Surface reaction rate|Boundaries 3–4|Meta, + operation|

#### Weak Expressions

|**Weak expression**|**Integration order**|**Integration frame**|**Selection**|
|-|-|-|-|
|tds2.J0\_M\*test(M)\*tds2.d|4|Spatial|Boundaries 3–4|
|tds2.J0\_Mas\*test(Mas)\*tds2.d|4|Spatial|Boundaries 3–4|
|tds2.J0\_Mat\*test(Mat)\*tds2.d|4|Spatial|Boundaries 3–4|

## Multiphysics

### Reacting Flow, Diluted Species 1

Used products

COMSOL Multiphysics

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: All domains|

#### Coupled Interfaces

Settings

|**Description**|**Value**|
|-|-|
|Fluid flow|Laminar Flow (spf) {spf}|
|Species transport|Transport of Diluted Species 2 (tds2) {tds2}|

#### Variables

|**Name**|**Expression**|**Description**|**Selection**|**Details**|
|-|-|-|-|-|
|tds2.cVar\_M|M|Species|Boundaries 1–8||
|tds2.cVar\_Mas|Mas|Species|Boundaries 1–8||
|tds2.cVar\_Mat|Mat|Species|Boundaries 1–8||
|spf.nuT|0|Turbulent kinematic viscosity|Domain 1||
|rfd1.ux|u|Velocity field, x-component|Domain 1||
|rfd1.uy|v|Velocity field, y-component|Domain 1||
|rfd1.uz|0|Velocity field, z-component|Domain 1||
|rfd1.p|p|Pressure|Domain 1||
|rfd1.pA|spf.pA|Absolute pressure|Domain 1||
|rfd1.D\_M|max(0.5\*(tds2.DF\_Mxx+tds2.DF\_Myy),eps)|Diffusion coefficient|Domain 1||
|rfd1.Sc\_M|spf.mu/(spf.rho\*max(rfd1.D\_M,eps))|Schmidt number|Domain 1||
|rfd1.ScT\_M|nojac(1/(0.5/rfd1.ScTinf\_M+0.3\*spf.muT\*rfd1.Sc\_M/(spf.mu\*sqrt(rfd1.ScTinf\_M))-(0.3\*spf.muT\*rfd1.Sc\_M/spf.mu)^2\*(1-exp(-3.3333333333333335\*spf.mu/(rfd1.Sc\_M\*max(spf.muT,eps)\*sqrt(rfd1.ScTinf\_M))))))|Turbulent Schmidt number|Domain 1||
|rfd1.ScTinf\_M|0.85|Turbulent Schmidt number at infinity|Domain 1||
|rfd1.D\_Mas|max(0.5\*(tds2.DF\_Masxx+tds2.DF\_Masyy),eps)|Diffusion coefficient|Domain 1||
|rfd1.Sc\_Mas|spf.mu/(spf.rho\*max(rfd1.D\_Mas,eps))|Schmidt number|Domain 1||
|rfd1.ScT\_Mas|nojac(1/(0.5/rfd1.ScTinf\_Mas+0.3\*spf.muT\*rfd1.Sc\_Mas/(spf.mu\*sqrt(rfd1.ScTinf\_Mas))-(0.3\*spf.muT\*rfd1.Sc\_Mas/spf.mu)^2\*(1-exp(-3.3333333333333335\*spf.mu/(rfd1.Sc\_Mas\*max(spf.muT,eps)\*sqrt(rfd1.ScTinf\_Mas))))))|Turbulent Schmidt number|Domain 1||
|rfd1.ScTinf\_Mas|0.85|Turbulent Schmidt number at infinity|Domain 1||
|rfd1.D\_Mat|max(0.5\*(tds2.DF\_Matxx+tds2.DF\_Matyy),eps)|Diffusion coefficient|Domain 1||
|rfd1.Sc\_Mat|spf.mu/(spf.rho\*max(rfd1.D\_Mat,eps))|Schmidt number|Domain 1||
|rfd1.ScT\_Mat|nojac(1/(0.5/rfd1.ScTinf\_Mat+0.3\*spf.muT\*rfd1.Sc\_Mat/(spf.mu\*sqrt(rfd1.ScTinf\_Mat))-(0.3\*spf.muT\*rfd1.Sc\_Mat/spf.mu)^2\*(1-exp(-3.3333333333333335\*spf.mu/(rfd1.Sc\_Mat\*max(spf.muT,eps)\*sqrt(rfd1.ScTinf\_Mat))))))|Turbulent Schmidt number|Domain 1||
|rfd1.ScTinf\_Mat|0.85|Turbulent Schmidt number at infinity|Domain 1||
|rfd1.u\_cmix|model.input.u\_cmi1|Velocity field, x-component|Global|Meta|
|rfd1.u\_cmiy|model.input.u\_cmi2|Velocity field, y-component|Global|Meta|
|rfd1.u\_cmiz|model.input.u\_cmi3|Velocity field, z-component|Global|Meta|
|rfd1.pA\_cmi|model.input.pA\_cmi|Absolute pressure|Global|Meta|

#### Shape functions

|**Name**|**Shape function**|**Description**|**Shape frame**|**Selection**|
|-|-|-|-|-|
|M|Lagrange (Quadratic)|Molar concentration, M|Spatial|No domains|
|M|Lagrange (Quadratic)|Molar concentration, M|Material|No domains|
|M|Lagrange (Quadratic)|Molar concentration, M|Geometry|No domains|
|M|Lagrange (Quadratic)|Molar concentration, M|Mesh|No domains|
|Mas|Lagrange (Quadratic)|Molar concentration, Mas|Spatial|No domains|
|Mas|Lagrange (Quadratic)|Molar concentration, Mas|Material|No domains|
|Mas|Lagrange (Quadratic)|Molar concentration, Mas|Geometry|No domains|
|Mas|Lagrange (Quadratic)|Molar concentration, Mas|Mesh|No domains|
|Mat|Lagrange (Quadratic)|Molar concentration, Mat|Spatial|No domains|
|Mat|Lagrange (Quadratic)|Molar concentration, Mat|Material|No domains|
|Mat|Lagrange (Quadratic)|Molar concentration, Mat|Geometry|No domains|
|Mat|Lagrange (Quadratic)|Molar concentration, Mat|Mesh|No domains|

#### Constraints

|**Constraint**|**Constraint force**|**Shape function**|**Selection**|**Details**|
|-|-|-|-|-|
|rfd1.mfwustr\_M-rfd1.mfwdstr\_M|-subst(root.comp1.spf.c\_ifan\_dstr,root.comp1.spf.c\_ifan\_d,down(test(M)),root.comp1.spf.c\_ifan\_u,up(test(M)))|Lagrange (Quadratic)|No domains|Elemental|
|rfd1.mfwustr\_Mas-rfd1.mfwdstr\_Mas|-subst(root.comp1.spf.c\_ifan\_dstr,root.comp1.spf.c\_ifan\_d,down(test(Mas)),root.comp1.spf.c\_ifan\_u,up(test(Mas)))|Lagrange (Quadratic)|No domains|Elemental|
|rfd1.mfwustr\_Mat-rfd1.mfwdstr\_Mat|-subst(root.comp1.spf.c\_ifan\_dstr,root.comp1.spf.c\_ifan\_d,down(test(Mat)),root.comp1.spf.c\_ifan\_u,up(test(Mat)))|Lagrange (Quadratic)|No domains|Elemental|

## Mesh 1

Mesh statistics

|**Description**|**Value**|
|-|-|
|Status|Complete mesh|
|Mesh vertices|5897|
|Triangles|11403|
|Edge elements|389|
|Vertex elements|8|
|Number of elements|11403|
|Minimum element quality|0.6498|
|Average element quality|0.9127|
|Element area ratio|0.16592|
|Mesh area|22.22|

### Dimensioni (size)

Settings

|**Description**|**Value**|
|-|-|
|Calibrate for|Fluid dynamics|
|Maximum element size|0.075|
|Minimum element size|5E-4|
|Curvature factor|0.2|
|Maximum element growth rate|1.05|
|Predefined size|Extremely fine|
|Custom element size|Custom|

### Dimensioni 1 (size1)

Selection

|||
|-|-|
|Geometric entity level|Boundary|
|Selection|Geometry geom1: Dimension 1: Boundaries 1–4, 6–7|

Settings

|**Description**|**Value**|
|-|-|
|Calibrate for|Fluid dynamics|
|Maximum element size|0.237|
|Minimum element size|0.00677|
|Curvature factor|0.3|
|Maximum element growth rate|1.13|
|Predefined size|Fine|

### Raffinamento ai vertici 1 (cr1)

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Geometry geom1: Dimension 2: Domain 1|

Settings

|**Description**|**Value**|
|-|-|
|Boundary selection|geom1, Geometry geom1: Dimension 1: Boundaries 1–4, 6–7|
|Corner selection|geom1, Geometry geom1: Dimension 0: No points|

### Triangolare non strutturata 1 (ftri1)

Selection

|||
|-|-|
|Geometric entity level|Domain|
|Selection|Remaining|

Settings

|**Description**|**Value**|
|-|-|
|Number of iterations|4|
|Maximum element depth to process|4|

Information

|**Description**|**Value**|
|-|-|
|Last build time|< 1 second|
|Built with|COMSOL 6.3.0.420 (win64), Mar 23, 2026, 12:46:25 PM|

# Study 1

Computation information

|||
|-|-|
|Computation time|47 h 32 min 6 s|

## Time Dependent

|**Times**|
|-|
|range(0,150,30000)|

Study settings

|**Description**|**Value**|
|-|-|
|Include geometric nonlinearity|Off|

Study settings

|**Description**|**Value**|
|-|-|
|Output times|{0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 2250, 2400, 2550, 2700, 2850, 3000, 3150, 3300, 3450, 3600, 3750, 3900, 4050, 4200, 4350, 4500, 4650, 4800, 4950, 5100, 5250, 5400, 5550, 5700, 5850, 6000, 6150, 6300, 6450, 6600, 6750, 6900, 7050, 7200, 7350, 7500, 7650, 7800, 7950, 8100, 8250, 8400, 8550, 8700, 8850, 9000, 9150, 9300, 9450, 9600, 9750, 9900, 10050, 10200, 10350, 10500, 10650, 10800, 10950, 11100, 11250, 11400, 11550, 11700, 11850, 12000, 12150, 12300, 12450, 12600, 12750, 12900, 13050, 13200, 13350, 13500, 13650, 13800, 13950, 14100, 14250, 14400, 14550, 14700, 14850, 15000, 15150, 15300, 15450, 15600, 15750, 15900, 16050, 16200, 16350, 16500, 16650, 16800, 16950, 17100, 17250, 17400, 17550, 17700, 17850, 18000, 18150, 18300, 18450, 18600, 18750, 18900, 19050, 19200, 19350, 19500, 19650, 19800, 19950, 20100, 20250, 20400, 20550, 20700, 20850, 21000, 21150, 21300, 21450, 21600, 21750, 21900, 22050, 22200, 22350, 22500, 22650, 22800, 22950, 23100, 23250, 23400, 23550, 23700, 23850, 24000, 24150, 24300, 24450, 24600, 24750, 24900, 25050, 25200, 25350, 25500, 25650, 25800, 25950, 26100, 26250, 26400, 26550, 26700, 26850, 27000, 27150, 27300, 27450, 27600, 27750, 27900, 28050, 28200, 28350, 28500, 28650, 28800, 28950, 29100, 29250, 29400, 29550, 29700, 29850, 30000}|

Physics and variables selection

|**Key**|**Solve for**|
|-|-|
|Laminar Flow (spf) {spf}|On|
|Transport of Diluted Species 9 (tds) {tds}|On|
|Transport of Diluted Species 2 (tds2) {tds2}|On|

Physics and variables selection

|**Feature**|**Solve for**|
|-|-|
|Reacting Flow, Diluted Species 1 (rfd1) {rfd1}|On|

Store in output

|**Interface**|**Output**|**Selection**|
|-|-|-|
|Laminar Flow (spf) {spf}|Physics controlled||
|Transport of Diluted Species 9 (tds) {tds}|Physics controlled||
|Transport of Diluted Species 2 (tds2) {tds2}|Physics controlled||
|Reacting Flow, Diluted Species 1 (rfd1) {rfd1}|||

Mesh selection

|**Component**|**Mesh**|
|-|-|
|Component 1|Mesh 1 {mesh1}|

### 

