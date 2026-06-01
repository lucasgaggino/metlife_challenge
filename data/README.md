# Dataset - Insurance Costs

## Descripción

Este dataset contiene información sobre costos de seguros médicos individuales facturados por compañías de seguros.

## Columnas

| Columna | Tipo | Descripción | Valores Posibles |
|---------|------|-------------|------------------|
| `age` | int | Edad del asegurado | 18-64 |
| `sex` | str | Género del asegurado | male, female |
| `bmi` | float | Índice de Masa Corporal | 15.96-53.13 |
| `children` | int | Número de dependientes cubiertos | 0-5 |
| `smoker` | str | Si el asegurado es fumador | yes, no |
| `region` | str | Área geográfica de cobertura | northeast, northwest, southeast, southwest |
| `charges` | float | **TARGET** - Costos médicos facturados | $1,121.87 - $63,770.43 |

## Estadísticas

- **Total de registros**: ~1,338
- **Valores nulos**: 0
- **Duplicados**: 1

## Fuente

Dataset público de Kaggle: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Notas

- Los datos están en dólares USD
- BMI (Body Mass Index) se calcula como: peso(kg) / altura(m)²
- El dataset es balanceado en cuanto a género y región