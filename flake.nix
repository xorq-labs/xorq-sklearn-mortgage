{
  description = "xorq-sklearn-mortgage flake using uv2nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      # Load a uv workspace from a workspace root.
      # Uv2nix treats all uv projects as workspace projects.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Create package overlay from workspace.
      overlay = workspace.mkPyprojectOverlay {
        # Prefer prebuilt binary wheels as a package source.
        # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
        # Binary wheels are more likely to, but may still require overrides for library dependencies.
        sourcePreference = "wheel"; # or sourcePreference = "sdist";
        # Optionally customise PEP 508 environment
        # environ = {
        #   platform_release = "5.10.65";
        # };
      };

      # Extend generated overlay with build fixups
      #
      # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
      # This is an additional overlay implementing build fixups.
      # See:
      # - https://pyproject-nix.github.io/uv2nix/FAQ.html
      pyprojectOverrides = final: prev: {
        # Build system dependencies for cityhash
        cityhash = prev.cityhash.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ final.resolveBuildSystem {
              setuptools = [];
              wheel = [];
            };
        });


        # Google packages
        google-crc32c = prev.google-crc32c.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ final.resolveBuildSystem {
              setuptools = [];
              wheel = [];
            };
        });

        # Data processing packages
        pyarrow = prev.pyarrow.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ [
              final.setuptools
              final.cython
              final.numpy
              pkgs.cmake
              pkgs.pkg-config
            ];
          buildInputs =
            (old.buildInputs or [])
            ++ [
              pkgs.arrow-cpp
              pkgs.lz4
            ];
        });

        # Machine learning packages
        xgboost = prev.xgboost.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ [pkgs.cmake]
            ++ final.resolveBuildSystem (
              pkgs.lib.listToAttrs (map (name: pkgs.lib.nameValuePair name []) ["hatchling"])
            );
        });

        # Custom package override for xorq
        xorq = prev.xorq.overrideAttrs (old: {
          src = ./dist/xorq-0.3.1-py3-none-any.whl;
          format = "wheel";

          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ final.resolveBuildSystem {
              setuptools = [];
              wheel = [];
            };
          buildInputs = (old.buildInputs or []) ++ [pkgs.openssl];
        });
        # Custom package override for quickgrove
        quickgrove = prev.xorq.overrideAttrs (old: {
          src = ./dist/quickgrove-0.1.4-cp312-cp312-linux_x86_64.whl;
          format = "wheel";
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ final.resolveBuildSystem {
              setuptools = [];
              wheel = [];
            };
          buildInputs = (old.buildInputs or []) ++ [pkgs.openssl];
        });

        # Network/RPC packages
        grpcio = prev.grpcio.overrideAttrs (old: {
          NIX_CFLAGS_COMPILE =
            (old.NIX_CFLAGS_COMPILE or "")
            + " -DTARGET_OS_OSX=1 -D_DARWIN_C_SOURCE"
            + " -I${pkgs.zlib.dev}/include"
            + " -I${pkgs.openssl.dev}/include"
            + " -I${pkgs.c-ares.dev}/include";

          NIX_LDFLAGS =
            (old.NIX_LDFLAGS or "")
            + " -L${pkgs.zlib.out}/lib -lz"
            + " -L${pkgs.openssl.out}/lib -lssl -lcrypto"
            + " -L${pkgs.c-ares.out}/lib -lcares";

          buildInputs =
            (old.buildInputs or [])
            ++ [
              pkgs.zlib
              pkgs.openssl
              pkgs.c-ares
            ];

          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ [
              pkgs.pkg-config
              pkgs.cmake
            ];

          # Environment variables for grpcio build
          GRPC_PYTHON_BUILD_SYSTEM_OPENSSL = "1";
          GRPC_PYTHON_BUILD_SYSTEM_ZLIB = "1";
          GRPC_PYTHON_BUILD_SYSTEM_CARES = "1";

          preBuild = ''
            export PYTHONPATH=${final.setuptools}/${python.sitePackages}:$PYTHONPATH
          '';
        });

        # Database packages
        duckdb = prev.duckdb.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ [
              final.setuptools
              final.pybind11
              final.wheel
              pkgs.cmake
            ];
          buildInputs = (old.buildInputs or []) ++ [pkgs.openssl];
        });

        psycopg2-binary = prev.psycopg2-binary.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ [
              final.setuptools
              final.wheel
              pkgs.postgresql.pg_config
              pkgs.postgresql
            ];
          buildInputs = (old.buildInputs or []) ++ [pkgs.openssl];
        });
      };

      # This example is only using x86_64-linux
      pkgs = nixpkgs.legacyPackages.x86_64-linux;

      # Use Python 3.12 from nixpkgs
      python = pkgs.python312;

      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]
          );

    in
    {
      # Package a virtual environment as our main application.
      #
      # Enable no optional dependencies for production build.
      packages.x86_64-linux.default = pythonSet.mkVirtualEnv "xorq-sklearn-mortgage-env" workspace.deps.default;

      # Make hello runnable with `nix run`
      apps.x86_64-linux = {
        default = {
          type = "app";
          program = "${self.packages.x86_64-linux.default}/bin/hello";
        };
      };

      # This example provides two different modes of development:
      # - Impurely using uv to manage virtual environments
      # - Pure development using uv2nix to manage virtual environments
      devShells.x86_64-linux = {
        # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
        # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
        impure = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
          ];
          env =
            {
              # Prevent uv from managing Python downloads
              UV_PYTHON_DOWNLOADS = "never";
              # Force uv to use nixpkgs Python interpreter
              UV_PYTHON = python.interpreter;
            }
            // lib.optionalAttrs pkgs.stdenv.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
            };
          shellHook = ''
            unset PYTHONPATH
          '';
        };

        # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
        # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
        #
        # This means that any changes done to your local files do not require a rebuild.
        #
        # Note: Editable package support is still unstable and subject to change.
        uv2nix =
          let
            # Create an overlay enabling editable mode for all local dependencies.
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              # Use environment variable
              root = "$REPO_ROOT";
              # Optional: Only enable editable for these packages
              # members = [ "xorq-sklearn-mortgage" ];
            };

            # Override previous set with our overrideable overlay.
            editablePythonSet = pythonSet.overrideScope (
              lib.composeManyExtensions [
                editableOverlay

                # Apply fixups for building an editable package of your workspace packages
                (final: prev: {
                  xorq-sklearn-mortgage = prev.xorq-sklearn-mortgage.overrideAttrs (old: {
                    # It's a good idea to filter the sources going into an editable build
                    # so the editable package doesn't have to be rebuilt on every change.
                    src = lib.fileset.toSource {
                      root = old.src;
                      fileset = lib.fileset.unions [
                        (old.src + "/pyproject.toml")
                        (old.src + "/README.md")
                        (old.src + "/src/xorq_sklearn_mortgage/__init__.py")
                      ];
                    };

                    # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                    #
                    # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                    # This behaviour is documented in PEP-660.
                    #
                    # With Nix the dependency needs to be explicitly declared.
                    nativeBuildInputs =
                      old.nativeBuildInputs
                      ++ final.resolveBuildSystem {
                        editables = [ ];
                      };
                  });

                })
              ]
            );

            # Build virtual environment, with local packages being editable.
            #
            # Enable all optional dependencies for development.
            virtualenv = editablePythonSet.mkVirtualEnv "xorq-sklearn-mortgage-dev-env" workspace.deps.all;

          in
          pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];

            env = {
              # Don't create venv using uv
              UV_NO_SYNC = "1";

              # Force uv to use nixpkgs Python interpreter
              UV_PYTHON = python.interpreter;

              # Prevent uv from downloading managed Python's
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              # Undo dependency propagation by nixpkgs.
              unset PYTHONPATH

              # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
      };
    };
}
