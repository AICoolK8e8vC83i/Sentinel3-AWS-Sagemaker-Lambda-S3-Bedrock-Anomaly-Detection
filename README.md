# Cloud-Native Maritime Anomaly Detection via Multi-Agent VLM Orchestration

## Abstract

This project implements a production-ready Computer Vision and Vision-Language Model (VLM) pipeline designed for planetary-scale maritime monitoring. The architecture utilizes a "Triangular Logic Flow" to bridge the gap between raw pixel detection and high-level crisis reasoning. By leveraging RF-DETR for NMS-free object detection and a dual-model consensus mechanism (Qwen-3 235B and Claude 4.6 Sonnet), the system identifies navigational anomalies such as the Suez Canal blockages (e.g. "The Ever Given Ship, the Suez Canal, and the Operations Blunder, April 2024," DOI: 10.13140/RG.2.2.19396.23689). The backend is built on a serverless, zero-framework AWS architecture, ensuring infinite scalability and enterprise-grade security via SSE-KMS and granular IAM policies.

---

## Problem Statement

Global maritime traffic monitoring at scale remains an unsolved operational challenge. The volume of vessel movement across strategic waterways — the Suez Canal, Strait of Malacca, Strait of Hormuz — means that human analysts cannot maintain continuous situational awareness. Existing automated systems rely on AIS (Automatic Identification System) transponder data, which is both voluntary and easily spoofed, and commercial satellite imagery pipelines tend to be closed, expensive, and latency-bound. When anomalies occur — a vessel grounding, a suspicious anchoring pattern, an unauthorized port entry — the gap between event and detection can be measured in hours.

The Ever Given incident in March 2021 demonstrated concretely what this lag costs: six days of blockage, $9.6 billion per day in delayed trade, and a recovery operation that could have been partially automated at the detection stage. This project treats that incident as a canonical benchmark and asks: can a cloud-native CV/VLM pipeline running on commodity satellite imagery detect the preconditions of such an event before the blockage fully materializes?

---

## Prior Work and Motivating Research

**On satellite-based maritime detection:**
Existing literature on SAR (Synthetic Aperture Radar) and multispectral vessel detection — including work from ESA's Sentinel-1/2 constellation and ICEYE — demonstrates that sub-10m resolution imagery is sufficient for large vessel classification, but multi-class detection across vessel types (tanker, bulk carrier, container, fishing, patrol) degrades significantly below 3m GSD (Ground Sampling Distance). Sentinel-2 imagery, used here, operates at 10m resolution in visible bands, which creates the core object size constraint discussed in the Limitations section.

**On object detection architecture selection:**
The choice of RF-DETR (Radio-Frequency Detection Transformer) over YOLOv12/v13 was deliberate. YOLO-family architectures rely on Non-Maximum Suppression (NMS) as a post-processing step to resolve overlapping bounding box candidates. In terrestrial driving datasets this works well because objects rarely overlap. Maritime scenarios break this assumption: vessels anchored in port, ships queuing in narrow channels, and multi-vessel incidents all produce dense, overlapping bounding box candidates that NMS incorrectly suppresses as duplicates. RF-DETR, as a DETR-family end-to-end detector, performs set prediction directly — eliminating NMS entirely — which is a material advantage for port and channel density scenarios. See: Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020) for the foundational argument; RF-DETR extends this with improved backbone efficiency for satellite imagery applications.

**On multi-model VLM consensus:**
Single-model VLM inference for high-stakes anomaly classification carries unacceptable false-positive rates in operational settings. The dual-model consensus design here — routing Qwen-3 235B outputs through Claude 4.6 Sonnet as an auditing layer — is informed by ensemble methods in classical ML (Dietterich, 2000) applied to generative reasoning chains. Qwen-3 235B handles the initial scene description and bounding box contextualization; Claude 4.6 Sonnet performs the anomaly classification decision, with disagreements surfaced as uncertainty flags rather than silent resolution. This mirrors the "Sentinel / Auditor" split common in agentic verification systems.

---

## Architecture Overview

The pipeline follows a Triangular Logic Flow across three stages:

**Stage 1 — Ingestion (AWS S3 + Lambda)**
Sentinel-2 imagery tiles are ingested via S3 event triggers into a serverless Lambda function. No persistent compute is provisioned at rest. SSE-KMS encryption is applied at the S3 layer; IAM policies enforce least-privilege access between pipeline components. The choice of native Boto3 over abstraction frameworks such as CrewAI or LangGraph is intentional: framework overhead introduces non-deterministic failure modes and version-dependency brittleness that are unacceptable in a pipeline where each function invocation costs real money and latency. Direct Boto3 calls give full control over retry logic, timeout behavior, and cold-start mitigation.

**Stage 2 — Detection (RF-DETR on SageMaker / Lambda Container)**
Preprocessed image tiles are passed to a containerized RF-DETR inference endpoint. The model outputs bounding boxes, class probabilities, and confidence scores without NMS post-processing. Detections are serialized and written to a DynamoDB results table, keyed by tile ID and timestamp, for downstream consumption.

**Stage 3 — Reasoning (Qwen-3 235B → Claude 4.6 Sonnet)**
Detection payloads are routed to Qwen-3 235B for initial scene contextualization — the model produces a structured description of detected objects, their spatial relationships, and any anomalous patterns relative to baseline traffic in that region. This output, together with the original image crop and bounding box overlay, is passed to Claude 4.6 Sonnet for anomaly classification and risk scoring. The Sonnet model acts as auditor: it either confirms, revises, or escalates the Qwen-3 assessment. Final classifications are written to a notification queue (SNS) for downstream alerting.

The "Triangular Logic" naming reflects the three-node handoff: raw pixel data enters at vertex one (detection), semantic scene understanding is constructed at vertex two (Qwen-3), and the crisis-level reasoning decision is made at vertex three (Claude 4.6 Sonnet). The triangle closes when the classification result is reconciled against the original detection bounding boxes for traceability.

**Latency and the Qwen-3 to Claude 4.6 Handoff**
The inter-model routing step carries a latency tax. Cold Lambda invocations add 800ms–2s depending on container size; if the target for end-to-end anomaly detection is under a defined threshold (e.g. 850ms for the handoff specifically), warm-start provisioned concurrency is the correct lever — not persistent streaming connections, which introduce statefulness that complicates the serverless model. The current implementation uses provisioned concurrency on the routing Lambda with a TTL-based keep-warm policy. Persistent streaming (e.g. a long-lived WebSocket or gRPC stream between Qwen-3 and Claude endpoints) is under evaluation for high-throughput tile batches but is not deployed in the current version.

---

## Limitations

**Object resolution at Sentinel-2 scale.** At 10m GSD, vessels shorter than approximately 50m occupy fewer than 5 pixels in the visible band. Small fishing vessels, patrol craft, and inflatable rigid-hull boats are below reliable detection threshold. The system is calibrated for large commercial vessels (container ships, tankers, bulk carriers) where the pixel footprint is sufficient for confident bounding box regression.

**Multi-class detection degrades with class count.** The current RF-DETR checkpoint is trained on a limited maritime vessel taxonomy. Expanding to a fine-grained class hierarchy — distinguishing, say, Panamax from post-Panamax container ships, or crude tankers from LNG carriers — requires significantly more labeled satellite imagery than is publicly available. Synthetic data augmentation (via GANs or diffusion-based approaches) is one path; transfer learning from aerial drone datasets is another, though domain shift between aerial and orbital imagery is non-trivial.

**Temporal context is not yet incorporated.** The current pipeline treats each tile as a static snapshot. Vessel behavior anomalies — stopping in an unusual location, moving in the wrong direction, loitering — require multi-temporal comparison. A future version should incorporate frame differencing or trajectory modeling over a rolling time window.

**Cloud cover occlusion.** Sentinel-2 is a passive optical sensor. Cloud cover over a target region can render entire tiles unusable. Fusion with Sentinel-1 SAR imagery, which is cloud-penetrating, is the standard mitigation but adds preprocessing complexity.

**Cost at planetary scale.** Lambda invocations, SageMaker inference endpoints, and LLM API calls each carry per-request costs. At the current architecture, scanning the full Suez Canal corridor at 10-minute intervals would require cost controls and tile-level prioritization logic not yet implemented.

---

## Next Steps For Maritime and Anomaly Detection

The immediate development priorities are: multi-temporal tile comparison for trajectory-based anomaly detection; Sentinel-1 SAR fusion for cloud-robust coverage; expansion of the vessel class taxonomy using synthetic data augmentation; and a formal latency benchmark of the warm-start Lambda versus persistent-stream approaches for the inter-model handoff. Longer term, the pipeline's Triangular Logic architecture is general enough to be retargeted to other remote sensing anomaly detection tasks — flood detection, wildfire perimeter mapping, unauthorized construction identification — with model checkpoint substitution and domain-specific prompt engineering for the VLM reasoning layer.

---

## Citation

If referencing the Ever Given case study framing: "The Ever Given Ship, the Suez Canal, and the Operations Blunder, April 2024." DOI: 10.13140/RG.2.2.19396.23689.

For the RF-DETR architectural motivation: Carion, N. et al. "End-to-End Object Detection with Transformers." ECCV 2020.

---

## License

Creative Commons (CC) (1.0) Universal licenses
