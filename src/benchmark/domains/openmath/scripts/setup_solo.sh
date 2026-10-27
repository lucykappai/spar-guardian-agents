#!/usr/bin/env bash
set -euo pipefail
cd {workdir}
chown root:root {workdir}
chmod 711 {workdir} /home
{agents}
rm -f {workdir}/.stage.tgz
