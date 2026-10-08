{% macro replace_selected_cycle_partitions() %}

    {% set years = var(
        'incremental_years',
        var('election_years', [])
    ) %}

    {% set election_types = var(
        'incremental_election_types',
        var('election_types', [])
    ) %}

    {% if is_incremental()
          and years | length > 0
          and election_types | length > 0 %}

        delete from {{ this }}
        where election_year in (
            {% for year in years %}
                {{ year }}{% if not loop.last %}, {% endif %}
            {% endfor %}
        )
        and election_type in (
            {% for election_type in election_types %}
                '{{ election_type | replace("'", "''") }}'
                {% if not loop.last %}, {% endif %}
            {% endfor %}
        )

    {% else %}

        select 1

    {% endif %}

{% endmacro %}
