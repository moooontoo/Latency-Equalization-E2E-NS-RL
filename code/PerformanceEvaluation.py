# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:35:50 2018

@author: tianz
"""

def get_total_link_resources(links):
    link_resources = 0
    for link in links:
        link_resources += link[2]

    return link_resources


def get_total_node_resources(nodes):
    node_resources = 0
    for node in nodes:
        node_resources += node

    return node_resources


def get_total_resources(nodes, links):
    node_resources = get_total_node_resources(nodes)
    link_resources = get_total_link_resources(links)
    resources = node_resources + link_resources

    return resources


def get_node_utilization(current_s_nodes, original_s_nodes):
    current_node_resources = get_total_node_resources(current_s_nodes)
    total_node_resources = get_total_node_resources(original_s_nodes)
    used_node_resources = total_node_resources - current_node_resources

    node_utilization = 0
    try:
        node_utilization = used_node_resources / total_node_resources
    except ZeroDivisionError:
        print('除0错误')

    return node_utilization


def get_link_utilization(current_s_links, original_s_links):
    current_link_resources = get_total_link_resources(current_s_links)
    total_link_resources = get_total_link_resources(original_s_links)
    used_link_resources = total_link_resources - current_link_resources

    link_utilization = 0
    try:
        link_utilization = used_link_resources / total_link_resources
    except ZeroDivisionError:
        print('除0错误')

    return link_utilization


def get_utilization(current_s_nodes, original_s_nodes, current_s_links, original_s_links):
    current_resources = get_total_resources(current_s_nodes, current_s_links)
    total_resources = get_total_resources(original_s_nodes, original_s_links)
    used_resources = total_resources - current_resources

    utilization = 0
    try:
        utilization = used_resources / total_resources
    except ZeroDivisionError:
        print('除0错误')

    return utilization


def get_revenue_cost_ratio(revenue, cost):
    try:
        revenue_cost_ratio = revenue / cost
    except ZeroDivisionError:
        revenue_cost_ratio = 0
    return revenue_cost_ratio
