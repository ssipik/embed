variable "image_repo" {
  type    = string
  default = "getregistry.azurecr.io/rag_embed"
}
variable "image_tag" {
  type = string
}

# Stage 5 (EMBED). Registered by bitbucket-pipelines.yml, started by Airflow after
# every parse batch of the run succeeded:
#   nomad job dispatch -meta run_id=<airflow run_id> rag-embed-template
# One job per run: the stage runs its own worker threads, sized to the healthy
# tei-embed replicas when it starts. To embed faster, scale tei-embed first:
#   nomad job scale tei-embed tei-embed 4
job "rag-embed-template" {
  datacenters = ["dc1"]
  type        = "batch"

  parameterized {
    # The run_id plan was given; embed processes that run's batches.
    meta_required = ["run_id"]
  }

  group "embed" {
    count = 1

    restart {
      attempts = 0
      mode     = "fail"
    }
    # Retries are Airflow's job, not Nomad's.
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "embed" {
      driver = "docker"

      config {
        image   = "${var.image_repo}:${var.image_tag}"
        command = "python"
        args    = ["-m", "rag_embed", "--run-id", "${NOMAD_META_run_id}"]
      }

      vault {}

      # Vault secrets + service addresses from Consul, rendered as env vars (same
      # pattern as rag-postgres.hcl), so the code only ever reads env vars.
      # - RAG_TEI_URL goes through traefik (plain HTTP entrypoint), which spreads
      #   requests over all healthy tei-embed replicas.
      # - RAG_TEI_REPLICAS counts those replicas once, at start; the stage sizes
      #   its worker count from it (settings.py). Scaling tei-embed during a run
      #   only takes effect on the next run.
      template {
        data        = <<EOT
{{- with secret "kv/data/secret/cluster/rag" }}
{{- $user := .Data.data.POSTGRES_USER }}{{ $pw := .Data.data.POSTGRES_PASSWORD }}
{{- range service "rag-postgres" }}
RAG_PG_DSN=postgresql+psycopg://{{ $user }}:{{ $pw | urlquery }}@{{ .Address }}:{{ .Port }}/rag_ingestion
{{- end }}
{{- end }}
{{- range service "qdrant" }}
RAG_QDRANT_URL=http://{{ .Address }}:{{ .Port }}
{{- end }}
RAG_TEI_URL=http://tei-embed.service.get
RAG_TEI_REPLICAS={{ service "tei-embed" | len }}
EOT
        destination = "secrets/rag.env"
        env         = true
        # A batch run must not be restarted by a secret rotation or a replica change mid-run.
        change_mode = "noop"
      }

      # The compute is in TEI; this job holds the worker threads, BM25 (in-process)
      # and one group of chunks per worker.
      resources {
        cpu    = 2000
        memory = 2048
      }
    }
  }
}
