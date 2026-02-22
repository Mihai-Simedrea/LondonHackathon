export interface HeatmapMeshPayload {
  source: string;
  units: string;
  vertices: number[][];
  faces: number[][];
}

export interface HeatmapMappingPayload {
  roi_mask: number[];
  left_weights: number[];
  right_weights: number[];
  anchors: Record<string, { label: string; xyz: number[] }>;
}

export interface DisplayBrainMappingPayload extends HeatmapMappingPayload {
  schema_version: number;
  asset_id: string;
  vertex_count: number;
  face_count?: number;
  mesh_source?: string;
  orientation?: {
    scale?: number;
    rotation_euler_xyz?: [number, number, number];
    translation?: [number, number, number];
  };
}

export interface HeatmapWindowPayload {
  sec: number;
  timestamp: number;
  left_raw_score: number;
  right_raw_score: number;
  pulse_quality: number;
  n_samples: number;
  left_red_z?: number;
  left_ir_z?: number;
  left_amb_z?: number;
  right_red_z?: number;
  right_ir_z?: number;
  right_amb_z?: number;
}

export interface CsvHeatmapResponse {
  schema_version: number;
  source: Record<string, unknown>;
  mesh: HeatmapMeshPayload;
  mapping: HeatmapMappingPayload;
  windows: HeatmapWindowPayload[];
  display_delay_sec: number;
  viewer_defaults?: Record<string, unknown>;
  disclaimer?: string;
  sample_rate_est_hz?: number;
}

export interface LiveStatusPayload {
  type?: 'fnirs_status' | 'fnirs_sample_stats';
  connected: boolean;
  streaming: boolean;
  mock_mode?: boolean;
  session_id?: string | null;
  sample_count?: number;
  sample_rate_est_hz?: number;
  buffer_seconds?: number;
  display_delay_sec?: number;
  last_error?: string | null;
  message?: string;
}

export interface LiveMeshInitPayload {
  type: 'fnirs_mesh_init';
  mesh: HeatmapMeshPayload;
  mapping: HeatmapMappingPayload;
  display_delay_sec: number;
  disclaimer?: string;
  viewer_defaults?: Record<string, unknown>;
}

export interface LiveHeatmapFramePayload {
  type: 'fnirs_heatmap_frame';
  sensor_now_ts: number;
  display_ts: number;
  display_delay_sec: number;
  window: HeatmapWindowPayload;
  qc: {
    sample_rate_est_hz: number;
    n_samples_buffered: number;
    window_n_samples: number;
  };
}

export interface LiveErrorPayload {
  type: 'fnirs_error';
  message: string;
}

export type FnirsLiveEvent = LiveStatusPayload | LiveMeshInitPayload | LiveHeatmapFramePayload | LiveErrorPayload;
