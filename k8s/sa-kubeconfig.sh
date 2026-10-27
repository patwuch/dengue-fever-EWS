#!/usr/bin/env bash
# Write a kubeconfig that acts as the snakemake ServiceAccount (k8s/rbac.yaml) of one cluster,
# for the controller container. Run it with an admin kubeconfig for that cluster.
#
#   k8s/sa-kubeconfig.sh <admin kubeconfig> <api server URL> <output> [duration]
#
#   tmu (monthly inference, CPU only):
#            KUBE_CONTEXT=tmu k8s/sa-kubeconfig.sh ~/.kube/k3s-tmu.yaml https://tmu-clhm-nuc15crhu5:6443 \
#                ~/.config/dengue-ews/kubeconfig
#   laptop (training, bundles, xAI; the only cluster with a GPU):
#            sudo k8s/sa-kubeconfig.sh /etc/rancher/k3s/k3s.yaml https://desktop-ijgfcqd:6443 \
#                ~/.config/dengue-ews/kubeconfig-laptop
#            (launch the controller with --add-host desktop-ijgfcqd:host-gateway: the name is on
#            the API server's certificate and resolves to the WSL host from inside the container)
#
# KUBE_CONTEXT picks the admin context when the admin kubeconfig holds several. The token
# expires after [duration] (default 720h); rerun the script to renew it.
set -euo pipefail

admin=$1 server=$2 out=$3 duration=${4:-720h}
k() { kubectl --kubeconfig "$admin" ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }

ca=$(k config view --raw --minify -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')
token=$(k -n dengue-ews create token snakemake --duration="$duration")

mkdir -p "$(dirname "$out")"
umask 077
cat > "$out" <<EOF
apiVersion: v1
kind: Config
clusters:
  - name: dengue-ews
    cluster:
      server: $server
      certificate-authority-data: $ca
users:
  - name: snakemake
    user:
      token: $token
contexts:
  - name: dengue-ews
    context: {cluster: dengue-ews, user: snakemake, namespace: dengue-ews}
current-context: dengue-ews
EOF
# Owned by whoever ran sudo, so the controller container (--user $(id -u)) can read it.
[ -n "${SUDO_USER:-}" ] && chown "$SUDO_USER:" "$out"
echo "Wrote $out (snakemake@dengue-ews on $server, token valid $duration)"
