"""
Graph view of CloudTrail activity, learned at training time and used by CloudTrailFeatures.

CloudTrail is a graph: identities connect to IPs, APIs, services and to the roles they assume.
Single-event features see one edge at a time ("first time alice used this IP"); the graph adds
relationships between identities:

  - role-assumption edges   who assumes which role, role-to-role chaining, cross-account hops
                            (lateral movement / privilege escalation)
  - IPs shared by identities how many identities normally come from an IP, and how many use it in
                            one hour (one attacker machine using several stolen keys)
  - peer communities        identities grouped by the services they use (Louvain community detection);
                            touching a service none of your peers use is more suspicious than one
                            they all use. Unknown identities are compared with everyone.

All tables are plain dicts of str -> int so the fitted model serializes safely (skops).
"""
import logging
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

# arn:aws:iam::<account>:role/<name>, arn:aws:sts::<account>:assumed-role/<name>/<session>
ACCOUNT_FROM_ARN = r'^arn:aws[\w-]*:(?:iam|sts)::(\d{12}):'

# Services used by more than this share of identities say nothing about who is similar to whom
COMMON_SERVICE_SHARE = 0.5
# Beyond this many identities, community detection is skipped and everyone is a peer
MAX_IDENTITIES_FOR_COMMUNITIES = 5000


def account_of(arns):
    return arns.str.extract(ACCOUNT_FROM_ARN, expand=False)


def identity_communities(actors, services, seed=42):
    """
    Groups identities that use similar services.

    Builds an identity-identity graph where the edge weight is the Jaccard similarity of the sets of
    services two identities use (ignoring services nearly everyone uses), then runs Louvain.
    Returns {actor: community id}.
    """
    used = pd.DataFrame({'actor': actors, 'service': services}).drop_duplicates()
    identities = used['actor'].unique()
    if len(identities) > MAX_IDENTITIES_FOR_COMMUNITIES:
        logger.warning("%d identities: skipping community detection, every identity is compared with all others",
                       len(identities))
        return {actor: 0 for actor in identities}

    share = used.groupby('service')['actor'].nunique() / max(len(identities), 1)
    distinctive = used[used['service'].map(share) <= COMMON_SERVICE_SHARE]
    service_sets = distinctive.groupby('actor')['service'].agg(frozenset).to_dict()

    graph = nx.Graph()
    graph.add_nodes_from(identities)
    by_service = distinctive.groupby('service')['actor'].agg(list)
    for members in by_service:
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if not graph.has_edge(a, b):
                    sa, sb = service_sets[a], service_sets[b]
                    graph.add_edge(a, b, weight=len(sa & sb) / len(sa | sb))

    communities = nx.community.louvain_communities(graph, weight='weight', seed=seed)
    return {actor: i for i, members in enumerate(communities) for actor in members}


def peer_service_counts(actors, services, membership):
    """
    For each (community, service), how many identities in the community used the service, plus
    community sizes and how many identities used each service overall.
    """
    used = pd.DataFrame({'actor': actors, 'service': services}).drop_duplicates()
    used['community'] = used['actor'].map(membership).astype(str)
    in_community = used.groupby(['community', 'service']).size()
    return ({f"{c}|{s}": int(n) for (c, s), n in in_community.items()},
            {str(c): int(n) for c, n in pd.Series(membership).astype(str).value_counts().items()},
            {str(s): int(n) for s, n in used.groupby('service')['actor'].nunique().items()})
