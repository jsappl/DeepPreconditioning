{
    if (correctPhi)
    {
        rAU.ref() = 1.0/UEqn.A();
    }
    else
    {
        rAU = 1.0/UEqn.A();
    }

    surfaceScalarField rAUf("rAUf", fvc::interpolate(rAU()));
    volVectorField HbyA(constrainHbyA(rAU()*UEqn.H(), U, p_rgh));
    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::flux(HbyA)
      + MRF.zeroFilter(fvc::interpolate(rho*rAU())*fvc::ddtCorr(U, phi, Uf))
    );
    MRF.makeRelative(phiHbyA);

    if (p_rgh.needReference())
    {
        fvc::makeRelative(phiHbyA, U);
        adjustPhi(phiHbyA, U, p_rgh);
        fvc::makeAbsolute(phiHbyA, U);
    }

    surfaceScalarField phig
    (
        (
            mixture.surfaceTensionForce()
          - ghf*fvc::snGrad(rho)
        )*rAUf*mesh.magSf()
    );

    phiHbyA += phig;

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF);

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix p_rghEqn
        (
            fvm::laplacian(rAUf, p_rgh) == fvc::div(phiHbyA)
        );

        p_rghEqn.setReference(pRefCell, getRefCellValue(p_rgh, pRefCell));
        p_rghEqn.solve();

        if (pimple.finalNonOrthogonalIter())
        {
            // Dump L,d of PPE Lp=d to disk, cf. https://is.gd/FMYQsn and https://is.gd/d6CE3q
            List<List<scalar>> L; // init 2-dimensional list L
            List<scalar> d; // init list d

            // Initialization of matrix/vector.
            L.resize(p_rgh.size());
            d.resize(p_rgh.size());
            forAll(L, i)
            {
                L[i].resize(p_rgh.size()); // every element of L is array
                forAll(L[i], j) // clearing L and d
                {
                    L[i][j] = 0.0;
                }
                d[i] = 0.0;
            }

            // Assigning diagonal coefficients and source term.
            forAll(p_rgh, i)
            {
                L[i][i] = p_rghEqn.diag()[i];
                d[i] = p_rghEqn.source()[i];
            }

            // Assigning off-diagonal coefficients.
            for (label faceI=0; faceI<p_rghEqn.lduAddr().lowerAddr().size(); faceI++)
            {
                label l = p_rghEqn.lduAddr().lowerAddr()[faceI];
                label u = p_rghEqn.lduAddr().upperAddr()[faceI];
                L[l][u] = p_rghEqn.upper()[faceI];
                L[u][l] = p_rghEqn.upper()[faceI];
            }

            // Assigning contribution from BC.
            forAll(p_rgh.boundaryField(), patchI)
            {
                const fvPatch &ptch = p_rgh.boundaryField()[patchI].patch();
                forAll(ptch, faceI)
                {
                    label cellI = ptch.faceCells()[faceI];
                    L[cellI][cellI] += p_rghEqn.internalCoeffs()[patchI][faceI];
                    d[cellI] += p_rghEqn.boundaryCoeffs()[patchI][faceI];
                }
            }

            FILE *fL = fopen("matrix.csv", "w");
            for (int i=0; i<p_rgh.size(); i++)
            {
                for (int j=0; j<p_rgh.size(); j++)
                {
                    if( L[i][j] != 0.0 ) {
                        fprintf(fL, "%i,%i,%.32f\n", i, j, L[i][j]);
                    }
                }
            }
            fclose(fL);

            FILE *fsolution = fopen("solution.csv", "w");
            for (int i=0; i<p_rgh.size(); i++)
            {
                fprintf(fsolution, "%.32f\n", p_rghEqn.psi()[i]);
            }
            fclose(fsolution);

            FILE *fd = fopen("right_hand_side.csv", "w");
            for (int i=0; i<p_rgh.size(); i++)
            {
                fprintf(fd, "%.32f\n", d[i]);
            }
            fclose(fd);
            // Dump matrix end.

            phi = phiHbyA - p_rghEqn.flux();

            p_rgh.relax();

            U = HbyA + rAU()*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf);
            U.correctBoundaryConditions();
            fvOptions.correct(U);
        }
    }

    #include "continuityErrs.H"

    // Correct Uf if the mesh is moving
    fvc::correctUf(Uf, U, phi);

    // Make the fluxes relative to the mesh motion
    fvc::makeRelative(phi, U);

    p == p_rgh + rho*gh;

    if (p_rgh.needReference())
    {
        p += dimensionedScalar
        (
            "p",
            p.dimensions(),
            pRefValue - getRefCellValue(p, pRefCell)
        );
        p_rgh = p - rho*gh;
    }

    if (!correctPhi)
    {
        rAU.clear();
    }
}
