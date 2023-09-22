import { useRef, useState } from 'react'
import { Button, Container, FileButton, Grid, Group, Image, Tabs, Text, Title } from "@mantine/core";
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { IconPhoto, IconPhotoFilled, IconPhotoHexagon, IconPhotoBolt } from '@tabler/icons-react';

import styles from './App.module.css'

const data = [
  {
    category: 'Bacterial Blight',
    probability: 80,
    highest: true,
  },
  {
    category: 'Tungro',
    probability: 8,
    highest: false,
  },
  {
    category: 'Brown Spot',
    probability: 2,
    highest: false,
  },
]

export default function App() {
  const [file, setFile] = useState(null);

  return (
    <Container className={styles.container_main}>
      <Title>Padi-CNN: Rice disease classification</Title>
      <Text size='sm'>MobileNetV2-based classifier with Grad-CAM visualization</Text>

      <Grid gutter="xl" className={styles.container_tabs}>
        <Grid.Col span={4}>
          <Title order={4}>Upload image</Title>
          <Group gap={"sm"} grow>
            <FileButton accept='image/png,image/jpeg' onChange={setFile}>
              {(props) => <Button {...props}>Upload</Button>}
            </FileButton>
            <Button variant='light'>Open sample</Button>
          </Group>

          {file && (
            <Text size="sm">
              Picked file: {file.name}
            </Text>
          )}

          <Title order={4} mt="lg">Classification results</Title>
          <Text ta="center">BACTERIAL BLIGHT</Text>

          <div style={{ height: "300px" }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} layout='vertical' margin={{ left: 20, right: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" label={{ value: 'Probability %', position: "bottom" }} />
                <YAxis dataKey="category" type='category' />
                <Tooltip />

                <Bar dataKey="probability">
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.highest ? '#8884d8' : '#82ca9d'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Grid.Col>
        <Grid.Col span={8}>
          <Title order={4}>Processed image</Title>
          <Tabs defaultValue="original" >
            <Tabs.List>
              <Tabs.Tab value="original" leftSection={<IconPhoto className={styles.tab_icon} />}>
                Original
              </Tabs.Tab>
              <Tabs.Tab value="superimposed" leftSection={<IconPhotoFilled className={styles.tab_icon} />}>
                Superimposed
              </Tabs.Tab>
              <Tabs.Tab value="clipped" leftSection={<IconPhotoHexagon className={styles.tab_icon} />}>
                Clipped
              </Tabs.Tab>
              <Tabs.Tab value="heatmap" leftSection={<IconPhotoBolt className={styles.tab_icon} />}>
                Heatmap
              </Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="original" p="md">
              <Image

                radius="md"
                src="https://images.unsplash.com/photo-1688920556232-321bd176d0b4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2370&q=80"
              />
            </Tabs.Panel>

            <Tabs.Panel value="superimposed">
              Messages tab content
            </Tabs.Panel>

            <Tabs.Panel value="clipped">
              Settings tab content
            </Tabs.Panel>

            <Tabs.Panel value="heatmap">
              Settings tab content
            </Tabs.Panel>
          </Tabs>
        </Grid.Col>
      </Grid>

    </Container>
  )
}
