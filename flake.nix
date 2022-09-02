{
  description = "Computational Neuroscience Environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      py3Opt = pkgs.python3.override {
        #        enableOptimizations = true;
        #        reproducibleBuild = false;
        self = py3Opt;
      };
      NEURON-src = pkgs.fetchurl {
          url = "https://github.com/neuronsimulator/nrn/releases/download/8.2.1/full-src-package-8.2.1.tar.gz";
          sha256 = "0kb0dn7nmivv3zflzkbj2fj3184zwp2crkxp0mdxkwm4kpnxqz0v";
        };
      NEURON = pkgs.stdenv.mkDerivation rec {
        pname = "NEURON";
        version = "8.2.1";
        outputs = ["out" "wheel"];
        propagatedNativeBuildInputs = (with pkgs; [
          mpi
          # TODO: X11 support
          # xquartz
          # xorg.libX11.dev
        ]);
        nativeBuildInputs = (with pkgs; [
          pkg-config
          readline81.dev
          cmake
          bison
          flex
          git
          py3Opt
          py3Opt.pkgs.wheel
          py3Opt.pkgs.setuptools
          py3Opt.pkgs.scikit-build
          py3Opt.pkgs.matplotlib
          py3Opt.pkgs.bokeh
          py3Opt.pkgs.ipython
          py3Opt.pkgs.cython
          py3Opt.pkgs.mpi4py
          py3Opt.pkgs.numpy
        ]) ++ pkgs.lib.optionals pkgs.stdenv.isDarwin (with pkgs; [
          xcbuild
        ]);
        configurePhase = ''
          sed -e 's#build/cmake_install#'"$out"'#'\
              -i setup.py
        '';
        buildPhase = ''
          python setup.py build_ext --disable-iv bdist_wheel
        '';
        # TODO: encode platform and python version
        installPhase = ''
          mkdir $out/dist
          mkdir $wheel
          cp dist/*.whl $wheel/
        '';
        src = NEURON-src;
      };
      NEURON-py = py3Opt.pkgs.buildPythonPackage rec {
        pname = "NEURON";
        version = "8.2.1";
        propagatedBuildInputs = [
          py3Opt.pkgs.setuptools
          py3Opt.pkgs.scikit-build
          py3Opt.pkgs.matplotlib
          py3Opt.pkgs.bokeh
          py3Opt.pkgs.ipython
          py3Opt.pkgs.cython
          py3Opt.pkgs.mpi4py
          py3Opt.pkgs.numpy
        ];
        format = "wheel";
        src = let
          dir = "${NEURON.wheel}/";
        in
          dir + (builtins.head (builtins.attrNames (builtins.readDir dir)));
      };
      mypy = py3Opt.withPackages (p: with p; [
        NEURON-py
        numpy
        scipy
        matplotlib
        seaborn
        pyyaml
        tqdm
        pathos
        prompt_toolkit
        python-lsp-server
      ]);
    in {
      devShell = pkgs.mkShell {
        nativeBuildInputs = [ pkgs.bashInteractive ];
        buildInputs = [
          mypy
          pkgs.readline81
        ];
      };
    });
}
