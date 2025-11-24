import sys
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from .models import GAT
from .training import Trainer
from .training.tensorborad_utils import log_district_visualizations_to_tensorboard
from .utils import get_logger

logger = get_logger(__name__)

def train_final_model(config, dataset, args):
    """Train final model on all data without validation split.

    Args:
        config: Training configuration
        dataset: BuildingGraphDataset
        args: Command line arguments
    """
    logger.info("=" * 80)
    logger.info("Final Model Training (Full Data, No Validation)")
    logger.info("=" * 80)

    # Load all data for training
    data_list = [dataset.get(i) for i in range(len(dataset))]
    logger.info("Training on all %d districts", len(data_list))

    # Initialize model
    logger.info("Initializing model...")
    model = GAT(
        in_features=dataset.num_features,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        negative_slope=config.negative_slope,
        add_self_loops=config.add_self_loops
    )

    logger.info("Model:\n%s", model)

    # Initialize trainer (no validation data)
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_data_list=data_list,
        val_data_list=None  # No validation in final training
    )

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.resume_from_checkpoint(Path(args.resume))

    # Train
    try:
        history = trainer.train()

        logger.info("Training completed successfully!")
        logger.info("Final model saved to: %s", Path(config.output_root_dir) / 'models')

        # === VISUALIZATION PHASE (after training completes) ===
        enable_final_visualization = getattr(config, 'enable_final_visualization', True)

        if enable_final_visualization and config.enable_tensorboard:
            logger.info("Loading best model and generating visualizations...")

            try:
                # Determine device
                device = torch.device(config.device if hasattr(config, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')

                # Load the best model checkpoint
                best_checkpoint_path = Path(config.checkpoint_dir) / "best.pt"
                if best_checkpoint_path.exists():
                    checkpoint = torch.load(best_checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Loaded best model from %s", best_checkpoint_path)
                else:
                    logger.warning("Best checkpoint not found at %s, using current model", best_checkpoint_path)

                model.to(device)
                model.eval()

                # Create TensorBoard writer for final training
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                tensorboard_dir = Path(config.log_dir) / f"{config.model_identifier}_final_{timestamp}"
                tensorboard_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(log_dir=str(tensorboard_dir))
                logger.info("TensorBoard logging to %s", tensorboard_dir)

                # Log training metrics to TensorBoard
                logger.info("Logging metrics to TensorBoard...")
                train_losses = history.get('train_loss', [])
                train_accs = history.get('train_acc', [])

                for epoch_idx, (train_loss, train_acc) in enumerate(zip(train_losses, train_accs), 1):
                    writer.add_scalar('train/loss', train_loss, epoch_idx)
                    writer.add_scalar('train/accuracy', train_acc, epoch_idx)

                # Generate visualizations for training data
                max_visualize = getattr(config, 'max_visualize_districts', 10)

                # === Save ALL visualizations to output directory ===
                logger.info("Generating and saving visualizations for ALL %d districts...", len(data_list))

                # Create visualization output directory
                vis_output_dir = Path(config.output_dir) / 'visualizations'
                vis_output_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Saving visualizations to: %s", vis_output_dir)

                try:
                    from .training.tensorborad_utils import visualize_district_predictions
                    import geopandas as gpd
                    import numpy as np
                    from PIL import Image

                    # Load building and district geometries once
                    buildings_all = gpd.read_file(Path(config.building_path))
                    districts_gdf = None
                    if hasattr(config, 'district_path') and Path(config.district_path).exists():
                        try:
                            districts_gdf = gpd.read_file(Path(config.district_path))
                            logger.info("Loaded district geometries")
                        except Exception as e:  # pylint: disable=broad-except
                            logger.warning("Failed to load district geometries: %s", e)

                    # Load adjacency directory if available
                    adjacency_dir = Path(config.adjacency_dir) if hasattr(config, 'adjacency_dir') else None

                    # Process all districts
                    saved_count = 0
                    failed_count = 0

                    with torch.no_grad():
                        for idx, data in enumerate(data_list):
                            try:
                                district_id = data.district_id if hasattr(data, 'district_id') else idx

                                # GAT predictions
                                data_device = data.to(device)
                                edge_attr = None
                                if hasattr(data_device, 'edge_attr') and data_device.edge_attr is not None:
                                    edge_attr = data_device.edge_attr.to(device)

                                if hasattr(model, 'forward_inference'):
                                    logits, embeddings = model.forward_inference(data_device.x, data_device.edge_index, edge_attr)
                                else:
                                    logits = model(data_device.x, data_device.edge_index, edge_attr)
                                    embeddings = None

                                gat_predictions = logits.argmax(dim=1).cpu().numpy()
                                ground_truth = data.y.cpu().numpy()

                                # Get district buildings
                                district_buildings = None
                                if districts_gdf is not None:
                                    try:
                                        district_geom = districts_gdf[districts_gdf['FID'] == district_id].geometry
                                        if len(district_geom) > 0:
                                            district_geom = district_geom.iloc[0]
                                            district_buildings = buildings_all[buildings_all.intersects(district_geom)].copy()
                                    except Exception:
                                        pass

                                if district_buildings is None or len(district_buildings) == 0:
                                    for field in ['district_id', 'FID', 'id', 'TAZ_ID']:
                                        if field in buildings_all.columns:
                                            district_buildings = buildings_all[buildings_all[field] == district_id].copy()
                                            if len(district_buildings) > 0:
                                                break

                                if district_buildings is None or len(district_buildings) == 0:
                                    logger.warning("No buildings found for district %s, skipping", district_id)
                                    failed_count += 1
                                    continue

                                # Match building counts
                                if len(district_buildings) != len(gat_predictions):
                                    min_len = min(len(district_buildings), len(gat_predictions))
                                    district_buildings = district_buildings.iloc[:min_len]
                                    gat_predictions = gat_predictions[:min_len]
                                    ground_truth = ground_truth[:min_len]

                                # Generate spectral predictions if available
                                spectral_predictions = None
                                spectral_clusters = None

                                if embeddings is not None and adjacency_dir is not None:
                                    try:
                                        import pandas as pd
                                        from .utils.spectral_clustering import perform_spectral_clustering_pipeline
                                        from .utils.feature_extractor import extract_clustering_features
                                        from .utils.graph_utils_ext import (
                                            extract_subgraph,
                                            extract_subgraph_from_adjacency,
                                            merge_component_results,
                                            get_connected_components_from_adjacency
                                        )

                                        adjacency_path = adjacency_dir / f"district_{district_id}_adjacency.pkl"
                                        if adjacency_path.exists():
                                            adjacency_matrix = pd.read_pickle(adjacency_path)
                                            building_ids = adjacency_matrix.index.tolist()

                                            clustering_features, _ = extract_clustering_features(
                                                district_buildings, scaler=None, fit_scaler=True
                                            )

                                            # Get spectral clustering config from config object
                                            spectral_config = {
                                                'embedding_weight': config.spectral_embedding_weight,
                                                'feature_weight': config.spectral_feature_weight,
                                                'distance_weight': config.spectral_distance_weight,
                                                'distance_scale': config.spectral_distance_scale,
                                                'use_confidence_weighted_voting': config.spectral_use_confidence_weighted_voting,
                                                'min_component_size': config.spectral_min_component_size,
                                                'min_cluster_size': config.spectral_min_cluster_size,
                                                'max_hops': config.spectral_max_hops,
                                                'oversample_factor': config.spectral_oversample_factor
                                            }

                                            # CRITICAL FIX: Connected component processing based on ADJACENCY MATRIX
                                            # (actual spatial Voronoi boundaries), not PyG edge_index
                                            component_labels_np, num_components = get_connected_components_from_adjacency(
                                                adjacency_matrix,
                                                building_ids
                                            )

                                            component_results = []
                                            for comp_id in range(num_components):
                                                comp_mask = (component_labels_np == comp_id)
                                                comp_size = comp_mask.sum()

                                                comp_data, _ = extract_subgraph(data_device, comp_mask, building_ids)

                                                comp_edge_attr = None
                                                if hasattr(comp_data, 'edge_attr') and comp_data.edge_attr is not None:
                                                    comp_edge_attr = comp_data.edge_attr.to(device)

                                                if hasattr(model, 'forward_inference'):
                                                    comp_logits, comp_embeddings = model.forward_inference(
                                                        comp_data.x, comp_data.edge_index, comp_edge_attr
                                                    )
                                                else:
                                                    comp_logits = model(comp_data.x, comp_data.edge_index, comp_edge_attr)
                                                    comp_embeddings = None

                                                comp_embeddings_np = comp_embeddings.cpu().numpy() if comp_embeddings is not None else None
                                                comp_logits_np = comp_logits.cpu().numpy()
                                                comp_gat_labels = comp_logits.argmax(dim=1).cpu().numpy()

                                                if comp_size >= spectral_config['min_component_size'] and comp_embeddings_np is not None:
                                                    comp_building_ids = [building_ids[i] for i in range(len(comp_mask)) if comp_mask[i]]
                                                    comp_clustering_features = clustering_features[comp_mask]
                                                    comp_adjacency = extract_subgraph_from_adjacency(adjacency_matrix, comp_building_ids)

                                                    comp_clusters, comp_final_labels, _, _, _ = perform_spectral_clustering_pipeline(
                                                        embeddings=comp_embeddings_np,
                                                        features=comp_clustering_features,
                                                        adjacency_matrix=comp_adjacency,
                                                        gat_labels=comp_gat_labels,
                                                        gat_logits=comp_logits_np,
                                                        building_ids=comp_building_ids,
                                                        voronoi_areas=None,
                                                        n_clusters=None,
                                                        use_confidence_weighted_voting=spectral_config['use_confidence_weighted_voting'],
                                                        embedding_weight=spectral_config['embedding_weight'],
                                                        feature_weight=spectral_config['feature_weight'],
                                                        distance_weight=spectral_config['distance_weight'],
                                                        distance_scale=spectral_config['distance_scale'],
                                                        min_cluster_size=spectral_config['min_cluster_size'],
                                                        max_hops=spectral_config['max_hops'],
                                                        oversample_factor=spectral_config['oversample_factor'],
                                                        random_state=42
                                                    )
                                                else:
                                                    comp_clusters = np.zeros(comp_size, dtype=int)
                                                    comp_final_labels = np.full(comp_size, 9, dtype=int)

                                                component_results.append({
                                                    'component_id': comp_id,
                                                    'num_nodes': comp_size,
                                                    'embeddings': comp_embeddings_np if comp_embeddings_np is not None else np.zeros((comp_size, embeddings.shape[1])),
                                                    'logits': comp_logits_np,
                                                    'gat_labels': comp_gat_labels,
                                                    'cluster_assignments': comp_clusters,
                                                    'final_labels': comp_final_labels,
                                                    'node_mask': comp_mask
                                                })

                                            merged = merge_component_results(component_results, component_labels_np)
                                            spectral_clusters = merged['cluster_assignments']
                                            spectral_predictions = merged['final_labels']
                                    except Exception as e:
                                        logger.debug("Spectral clustering failed for district %s: %s", district_id, e)

                                # Generate visualization
                                num_classes = int(ground_truth.max()) + 1
                                image_array = visualize_district_predictions(
                                    district_id=district_id,
                                    buildings_gdf=district_buildings,
                                    predictions=gat_predictions,
                                    ground_truth=ground_truth,
                                    num_classes=num_classes,
                                    spectral_predictions=spectral_predictions,
                                    spectral_clusters=spectral_clusters
                                )

                                # Save to file
                                output_path = vis_output_dir / f"district_{district_id}.png"
                                image = Image.fromarray(image_array)
                                image.save(output_path, format='PNG', dpi=(300, 300))
                                saved_count += 1

                                if (saved_count) % 10 == 0:
                                    logger.info("Saved %d/%d visualizations...", saved_count, len(data_list))

                            except Exception as e:
                                logger.warning("Failed to generate visualization for district %s: %s", 
                                             district_id if 'district_id' in locals() else idx, e)
                                failed_count += 1

                    logger.info("Completed saving visualizations: %d saved, %d failed", saved_count, failed_count)

                except Exception as e:
                    logger.error("Failed to generate and save visualizations: %s", e, exc_info=True)

                # === Upload subset to TensorBoard ===
                logger.info("Uploading %d visualizations to TensorBoard...", min(max_visualize, len(data_list)))

                try:
                    log_district_visualizations_to_tensorboard(
                        writer=writer,
                        model=model,
                        data_list=data_list[:max_visualize],
                        building_path=Path(config.building_path),
                        epoch=0,  # Use 0 since this is post-training
                        tag='train_final',
                        max_districts=max_visualize,
                        device=str(device),
                        district_path=Path(config.district_path) if hasattr(config, 'district_path') else None,
                        adjacency_dir=Path(config.adjacency_dir) if hasattr(config, 'adjacency_dir') else None,
                        enable_spectral_clustering=True
                    )
                    logger.info("Completed TensorBoard visualizations")
                except Exception as e:
                    logger.error("Failed to upload TensorBoard visualizations: %s", e, exc_info=True)

                # Add text summary
                text_summary = "**Final Model Training**\n\n"
                text_summary += f"- Training Districts: {len(data_list)}\n"
                text_summary += f"- Total Epochs: {len(train_losses)}\n"
                if train_accs:
                    text_summary += f"- Final Train Acc: {train_accs[-1]:.4f}\n"
                text_summary += f"\n**Model Configuration:**\n\n"
                text_summary += f"- Hidden Dim: {config.hidden_dim}\n"
                text_summary += f"- Num Layers: {config.num_layers}\n"
                text_summary += f"- Num Heads: {config.num_heads}\n"
                text_summary += f"- Dropout: {config.dropout}\n"

                writer.add_text('training/summary', text_summary, 0)

                # Close writer
                writer.flush()
                writer.close()
                logger.info("Closed TensorBoard writer - logs saved to %s", tensorboard_dir)

            except Exception as exc:
                logger.error("Visualization failed: %s", exc, exc_info=True)
        else:
            if not enable_final_visualization:
                logger.info("Visualization disabled by configuration")
            else:
                logger.info("TensorBoard disabled, skipping visualization")

        return history

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
